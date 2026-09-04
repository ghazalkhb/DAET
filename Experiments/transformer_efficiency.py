# ==============================================================================
# transformer_efficiency.py
# Transformer Training Cost, Inference Runtime, and Memory Consumption.
#
# This script measures, for the compact Transformer used in Section 5.3
# of the paper, and for each classical detector, per-signature:
#   - Training wall-clock time (Transformer, ARIMA only)
#   - Approx. process RSS delta during training  (psutil, before/after)
#   - Per-step inference wall-clock time (all methods)
#   - Approx. process RSS delta during inference  (psutil, before/after)
#   - Model parameter count (Transformer)
#
# Hyperparameters used match the paper's tuned values:
#   Firefox-Android : Transformer(window=20, d_model=16), SMA(w=40), EWMA(α=0.05)
#   mozilla-beta    : Transformer(window=10, d_model=32), SMA(w=20), EWMA(α=0.05)
#
# To keep total runtime tractable, measurements are collected on the first
# MAX_SIGS_PER_DATASET signatures per dataset (ranked by descending series
# length, consistent with run_full_evaluation.py ordering).  Totals over all
# signatures are then extrapolated to the full dataset count.
#
# Outputs (under Experiments/results/):
#   transformer_efficiency_detail.csv   — per-signature timing/memory
#   transformer_efficiency_summary.json — aggregate statistics + totals
#   plots/efficiency_training_time.png
#   plots/efficiency_inference_time.png
#   plots/efficiency_memory.png
#   plots/efficiency_radar.png
#
# Usage:
#   python "Experiments/transformer_efficiency.py"
# ==============================================================================

import os, sys, re, glob, json, math, time, warnings
warnings.filterwarnings("ignore")
import psutil as _psutil
_PROC = _psutil.Process()

def _rss_kb():
    """Current process RSS in KB."""
    return _PROC.memory_info().rss / 1024.0

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("WARNING: PyTorch not found. Transformer measurements will be skipped.")
    sys.exit(1)

# ==============================================================================
#  Paths
# ==============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(ROOT, "Data")
TS_BASE     = os.path.join(DATA_DIR, "timeseries-data")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================================================================
#  Experiment parameters — mirror run_full_evaluation.py exactly
# ==============================================================================
BETA              = 0.30
THRESHOLD_ANOMALY = 0.30
K_SIGMA           = 3.0
THR_WINDOW        = 30
MIN_THRESHOLD     = 1e-6
ARIMA_ORDER       = (1, 1, 1)
ARIMA_MIN_HISTORY = 20
ARIMA_WINDOW      = 60
TRAIN_RATIO       = 0.60
VAL_RATIO         = 0.20

# Compact Transformer fixed training hypers
TRANS_NHEAD        = 2
TRANS_NLAYERS      = 1
TRANS_DIM_FF_RATIO = 2
TRANS_DROPOUT      = 0.0
TRANS_LR           = 1e-2
TRANS_EPOCHS       = 20
TRANS_BATCH        = 256
TRANS_PATIENCE     = 6
MEASUREMENT_SEED   = 42

# Per-dataset tuned HP (from paper / combined_summary.json)
DATASET_HP = {
    "firefox-android": {"window": 20, "d_model": 16, "sma_w": 40, "ewma_alpha": 0.05},
    "mozilla-beta":    {"window": 10, "d_model": 32, "sma_w": 20, "ewma_alpha": 0.05},
}

# Maximum signatures to measure per dataset for efficiency experiment
# (set to None to measure all — warning: mozilla-beta has 1477 sigs)
MAX_SIGS_PER_DATASET = {
    "firefox-android": None,   # 342 sigs — measure all
    "mozilla-beta":    200,    # subsample first 200 (longest) sigs
}

DATASETS = [
    ("firefox-android", "firefox-android", "firefox-android"),
    ("mozilla-beta",    "mozilla-beta",    "mozilla-beta"),
]

# ==============================================================================
#  Utility helpers (mirrors run_full_evaluation.py)
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


# ==============================================================================
#  Compact Transformer definition (identical to run_full_evaluation.py)
# ==============================================================================

class TSTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, window):
        super().__init__()
        self.window   = window
        self.proj_in  = nn.Linear(1, d_model)
        enc_layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=TRANS_DROPOUT, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=TRANS_NLAYERS)
        self.proj_out = nn.Linear(d_model, 1)

    def forward(self, x):    # x: (B, window, 1)
        h = self.proj_in(x)
        h = self.encoder(h)
        return self.proj_out(h[:, -1, :]).squeeze(-1)


def count_parameters(model):
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
#  Measured training function: returns (model_tuple, train_time_s, rss_delta_KB)
# ==============================================================================

def train_transformer_measured(y_train, window, d_model, seed):
    """
    Train the compact Transformer on y_train.
    Returns (model_tuple, train_time_s, rss_delta_KB, n_epochs_run, n_params).
    rss_delta_KB = approx. process RSS delta (psutil before/after); NOT true peak.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    nhead  = min(TRANS_NHEAD, d_model)
    dim_ff = d_model * TRANS_DIM_FF_RATIO

    if len(y_train) <= window + 1:
        return None, 0.0, 0.0, 0, 0

    # --- Start measuring (before all allocations) ---
    rss_before = _rss_kb()
    t_start = time.perf_counter()

    y_t  = torch.tensor(y_train, dtype=torch.float32)
    mu   = y_t.mean(); sig = y_t.std()
    sig  = sig if sig > 1e-8 else torch.tensor(1.0)
    y_n  = (y_t - mu) / sig

    X = torch.stack([y_n[i: i + window] for i in range(len(y_n) - window)])
    Y = y_n[window:]
    X = X.unsqueeze(-1)

    ds   = torch.utils.data.TensorDataset(X, Y)
    dl   = torch.utils.data.DataLoader(ds, batch_size=TRANS_BATCH, shuffle=True)

    model   = TSTransformer(d_model, nhead, dim_ff, window).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=TRANS_LR)
    loss_fn = nn.MSELoss()
    n_params = count_parameters(model)

    best_loss = float("inf"); patience_ct = 0; best_state = None
    n_epochs_run = 0

    for ep in range(TRANS_EPOCHS):
        n_epochs_run += 1
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds)
        if ep_loss < best_loss - 1e-6:
            best_loss = ep_loss; patience_ct = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= TRANS_PATIENCE:
                break

    t_end   = time.perf_counter()
    rss_after = _rss_kb()
    # --- End measuring ---

    train_time_s  = t_end - t_start
    rss_delta_KB  = max(0.0, rss_after - rss_before)  # approx.; NOT true peak

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return (model, mu.item(), sig.item(), window), train_time_s, rss_delta_KB, n_epochs_run, n_params


# ==============================================================================
#  Measured inference function: returns (decisions, infer_time_s, rss_delta_KB)
# ==============================================================================

def infer_transformer_measured(y_history, y_segment, model_tuple):
    """Run transformer inference on y_segment; return (decisions, total_infer_s, rss_delta_KB)."""
    if model_tuple is None:
        return [0] * len(y_segment), 0.0, 0.0

    model, mu, sig, window = model_tuple
    combined   = np.concatenate([y_history, y_segment])
    offset     = len(y_history)
    combined_n = (combined.astype(np.float32) - mu) / max(sig, 1e-8)
    decisions  = []
    dh, sc     = [], 0.0

    rss_before = _rss_kb()
    t_start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for i_t in range(len(y_segment)):
            abs_i = offset + i_t
            act   = safe_float(y_segment[i_t])
            if act is None or abs_i < window:
                decisions.append(0)
                continue
            buf  = torch.tensor(combined_n[abs_i - window: abs_i],
                                dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pr   = model(buf).item() * max(sig, 1e-8) + mu
            delta = abs(act - pr)
            is_a  = 1 if delta > dyn_thr(dh) else 0
            sc    = update_score(sc, is_a)
            dh.append(delta)
            decisions.append(1 if sc >= THRESHOLD_ANOMALY else 0)

    t_end   = time.perf_counter()
    rss_after = _rss_kb()

    return decisions, t_end - t_start, max(0.0, rss_after - rss_before)


# ==============================================================================
#  Measured ARIMA fit and inference
# ==============================================================================

def arima_fit_measured(y_train):
    """Fit ARIMA on train set; return (fit, fit_time_s, rss_delta_KB) or (None, ...)."""
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    fit = None
    try:
        m   = StatsARIMA(np.array(y_train, dtype=float), order=ARIMA_ORDER,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = m.fit()
    except Exception:
        pass
    t1 = time.perf_counter()
    rss_after = _rss_kb()
    return fit, t1 - t0, max(0.0, rss_after - rss_before)


def arima_rolling_infer_measured(arima_fit, y_seed, y_segment):
    """Rolling ARIMA predictions; return (preds, total_infer_s, rss_delta_KB)."""
    params = arima_fit.params if arima_fit is not None else None
    buf    = list(np.array(y_seed[-ARIMA_WINDOW:], dtype=float))
    preds  = []

    rss_before = _rss_kb()
    t0 = time.perf_counter()

    for val in y_segment:
        pr = None
        if params is not None and len(buf) >= ARIMA_MIN_HISTORY:
            try:
                m   = StatsARIMA(np.array(buf, dtype=float), order=ARIMA_ORDER,
                                 enforce_stationarity=False, enforce_invertibility=False)
                res = m.filter(params)
                pr  = safe_float(res.forecast(1)[0])
            except Exception:
                pass
        if pr is None:
            pr = safe_float(buf[-1])
        preds.append(pr)
        buf.append(float(val))
        if len(buf) > ARIMA_WINDOW: buf.pop(0)

    t1 = time.perf_counter()
    rss_after = _rss_kb()
    return np.array(preds), t1 - t0, max(0.0, rss_after - rss_before)


# ==============================================================================
#  Classical inference (SMA/EWMA/LAST) measured
# ==============================================================================

def classical_infer_measured(y_history, y_segment, method, param):
    """Run one classical method on y_segment; return (decisions, infer_s, rss_delta_KB)."""
    sma_w  = param if method == "sma"  else 20
    alpha  = param if method == "ewma" else 0.30
    # Initialise EWMA from last of history
    ewma   = float(y_history[-1]) if len(y_history) > 0 else 0.0
    dh, sc = [], 0.0
    decisions = []

    rss_before = _rss_kb()
    t0 = time.perf_counter()

    for i_t, val in enumerate(y_segment):
        act = safe_float(val)
        if act is None:
            decisions.append(0)
            continue
        if method == "last":
            pr = safe_float(y_segment[i_t - 1] if i_t > 0 else y_history[-1])
        elif method == "sma":
            buf = np.concatenate([y_history, y_segment[:i_t]])[-sma_w:]
            pr  = safe_float(np.mean(buf)) if len(buf) > 0 else None
        elif method == "ewma":
            pr   = ewma
            ewma = alpha * act + (1 - alpha) * ewma
        else:
            pr = None
        if pr is None:
            decisions.append(0)
            continue
        delta = abs(act - pr)
        is_a  = 1 if delta > dyn_thr(dh) else 0
        sc    = update_score(sc, is_a)
        dh.append(delta)
        decisions.append(1 if sc >= THRESHOLD_ANOMALY else 0)

    t1 = time.perf_counter()
    rss_after = _rss_kb()
    return decisions, t1 - t0, max(0.0, rss_after - rss_before)


# ==============================================================================
#  Data loading (mirrors run_full_evaluation.py)
# ==============================================================================

def load_timeseries(ts_dir, alerts_df, repo_name, max_sigs=None):
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

    results.sort(key=lambda x: -len(x[1]))   # longest first
    if max_sigs is not None:
        results = results[:max_sigs]
    return results


# ==============================================================================
#  Main measurement loop
# ==============================================================================

def measure_dataset(dataset_name, repo_name, ts_dir, max_sigs):
    hp   = DATASET_HP[dataset_name]
    win  = hp["window"]
    dmod = hp["d_model"]
    sma_w     = hp["sma_w"]
    ewma_alpha= hp["ewma_alpha"]

    print(f"\n{'='*70}")
    print(f"  Measuring: {dataset_name}  (max_sigs={max_sigs})")
    print(f"  Transformer HP: window={win}, d_model={dmod}")
    print(f"  SMA w={sma_w}, EWMA alpha={ewma_alpha}")
    print(f"{'='*70}")

    alerts_df = pd.read_csv(ALERTS_CSV)
    sigs_data = load_timeseries(ts_dir, alerts_df, repo_name, max_sigs)
    n_loaded  = len(sigs_data)
    # count total sigs in this dataset (without cap)
    all_data  = load_timeseries(ts_dir, alerts_df, repo_name, None)
    n_total   = len(all_data)

    print(f"  Measuring {n_loaded} / {n_total} total signatures")

    # Parameter count (one model)
    nhead  = min(TRANS_NHEAD, dmod)
    dim_ff = dmod * TRANS_DIM_FF_RATIO
    dummy  = TSTransformer(dmod, nhead, dim_ff, win)
    n_params = count_parameters(dummy)
    print(f"  Compact Transformer: {n_params} parameters  "
          f"(~{n_params * 4 / 1024:.1f} KB model weights)")

    rows = []
    for i_sig, (sig, y, _) in enumerate(sigs_data):
        T        = len(y)
        T_train  = max(ARIMA_MIN_HISTORY + 2, int(round(TRAIN_RATIO * T)))
        T_val    = max(2, int(round(VAL_RATIO * T)))
        T_test   = T - T_train - T_val
        if T_test < 2:
            continue

        y_train = y[:T_train]
        y_val   = y[T_train: T_train + T_val]
        y_test  = y[T_train + T_val:]
        y_trainval = np.concatenate([y_train, y_val])

        row = dict(
            dataset       = dataset_name,
            signature_id  = sig,
            T_total       = T,
            T_train       = T_train,
            T_val         = T_val,
            T_test        = T_test,
            n_params      = n_params,
        )

        # ── Transformer training ──────────────────────────────────────────
        mt, tr_time, tr_mem, n_ep, _ = train_transformer_measured(
            y_train, win, dmod, MEASUREMENT_SEED)
        row["trans_train_time_s"]   = round(tr_time, 4)
        row["trans_train_mem_KB"]   = round(tr_mem,  2)
        row["trans_n_epochs"]       = n_ep

        # ── Transformer inference (test set) ─────────────────────────────
        if mt is not None:
            _, inf_time, inf_mem = infer_transformer_measured(y_trainval, y_test, mt)
            row["trans_infer_time_s"]   = round(inf_time, 6)
            row["trans_infer_time_ms_per_step"] = round(
                (inf_time / T_test) * 1000.0, 4) if T_test > 0 else None
            row["trans_infer_mem_KB"]   = round(inf_mem,  2)
        else:
            row["trans_infer_time_s"]   = None
            row["trans_infer_time_ms_per_step"] = None
            row["trans_infer_mem_KB"]   = None

        # ── ARIMA fit ─────────────────────────────────────────────────────
        arima_fit, fit_time, fit_mem = arima_fit_measured(y_train)
        row["arima_fit_time_s"]  = round(fit_time, 4)
        row["arima_fit_mem_KB"]  = round(fit_mem,  2)

        # ── ARIMA rolling inference (test set, after advancing through val) ─
        if arima_fit is not None:
            # advance through val first (mirrors run_full_evaluation.py)
            arima_fit2, _, _ = arima_fit_measured(y_trainval)  # refit on train+val
            _, rinf_time, rinf_mem = arima_rolling_infer_measured(
                arima_fit2, y_trainval, y_test)
            row["arima_infer_time_s"]  = round(rinf_time, 4)
            row["arima_infer_time_ms_per_step"] = round(
                (rinf_time / T_test) * 1000.0, 3) if T_test > 0 else None
            row["arima_infer_mem_KB"]  = round(rinf_mem,  2)
        else:
            row["arima_infer_time_s"]  = None
            row["arima_infer_time_ms_per_step"] = None
            row["arima_infer_mem_KB"]  = None

        # ── SMA inference ─────────────────────────────────────────────────
        _, sma_time, sma_mem = classical_infer_measured(
            y_trainval, y_test, "sma", sma_w)
        row["sma_infer_time_s"]   = round(sma_time, 6)
        row["sma_infer_time_ms_per_step"] = round(
            (sma_time / T_test) * 1000.0, 4) if T_test > 0 else None
        row["sma_infer_mem_KB"]   = round(sma_mem,  2)

        # ── EWMA inference ────────────────────────────────────────────────
        _, ewma_time, ewma_mem = classical_infer_measured(
            y_trainval, y_test, "ewma", ewma_alpha)
        row["ewma_infer_time_s"]  = round(ewma_time, 6)
        row["ewma_infer_time_ms_per_step"] = round(
            (ewma_time / T_test) * 1000.0, 4) if T_test > 0 else None
        row["ewma_infer_mem_KB"]  = round(ewma_mem,  2)

        # ── LAST inference ────────────────────────────────────────────────
        _, last_time, last_mem = classical_infer_measured(
            y_trainval, y_test, "last", None)
        row["last_infer_time_s"]  = round(last_time, 6)
        row["last_infer_time_ms_per_step"] = round(
            (last_time / T_test) * 1000.0, 4) if T_test > 0 else None
        row["last_infer_mem_KB"]  = round(last_mem,  2)

        rows.append(row)

        if (i_sig + 1) % 20 == 0 or (i_sig + 1) == n_loaded:
            print(f"  [{i_sig+1}/{n_loaded}]  sig={sig}  "
                  f"T={T}  "
                  f"trans_train={tr_time:.2f}s  "
                  f"arima_fit={fit_time:.2f}s")

    df = pd.DataFrame(rows)
    return df, n_total, n_params


# ==============================================================================
#  Aggregate summary per dataset
# ==============================================================================

def aggregate(df, n_total, n_measured, dataset_name, hp):
    def safe_mean(col):
        v = df[col].dropna()
        return float(v.mean()) if len(v) > 0 else None
    def safe_std(col):
        v = df[col].dropna()
        return float(v.std()) if len(v) > 1 else None
    def safe_sum(col):
        v = df[col].dropna()
        return float(v.sum()) if len(v) > 0 else None

    # Extrapolated total = mean_per_sig × n_total
    def extrap_total(col):
        m = safe_mean(col)
        return round(m * n_total, 2) if m is not None else None

    n_params = int(df["n_params"].iloc[0]) if len(df) > 0 else 0

    return {
        "dataset":             dataset_name,
        "n_total_sigs":        n_total,
        "n_measured_sigs":     n_measured,
        "transformer_hp":      hp,
        "n_params":            n_params,
        "model_weight_KB":     round(n_params * 4 / 1024.0, 2),

        # --- Training ---
        "trans_train": {
            "mean_per_sig_s":    round(safe_mean("trans_train_time_s") or 0, 4),
            "std_per_sig_s":     round(safe_std("trans_train_time_s")  or 0, 4),
            "mean_epochs":       round(safe_mean("trans_n_epochs")     or 0, 2),
            "mean_rss_delta_KB": round(safe_mean("trans_train_mem_KB") or 0, 2),  # approx. process RSS delta
            "std_rss_delta_KB":  round(safe_std("trans_train_mem_KB")  or 0, 2),
            "total_measured_s":  round(safe_sum("trans_train_time_s")  or 0, 2),
            "extrap_total_s":    extrap_total("trans_train_time_s"),
        },
        # --- Inference ---
        "trans_infer": {
            "mean_per_sig_s":    round(safe_mean("trans_infer_time_s") or 0, 6),
            "mean_ms_per_step":  round(safe_mean("trans_infer_time_ms_per_step") or 0, 4),
            "std_ms_per_step":   round(safe_std("trans_infer_time_ms_per_step")  or 0, 4),
            "mean_rss_delta_KB": round(safe_mean("trans_infer_mem_KB") or 0, 2),
        },
        # --- ARIMA ---
        "arima": {
            "mean_fit_time_s":   round(safe_mean("arima_fit_time_s")   or 0, 4),
            "mean_fit_mem_KB":   round(safe_mean("arima_fit_mem_KB")   or 0, 2),
            "mean_ms_per_step":  round(safe_mean("arima_infer_time_ms_per_step") or 0, 3),
            "mean_infer_mem_KB": round(safe_mean("arima_infer_mem_KB") or 0, 2),
            "total_measured_infer_s": round(safe_sum("arima_infer_time_s") or 0, 2),
            "extrap_infer_s":    extrap_total("arima_infer_time_s"),
        },
        # --- SMA ---
        "sma": {
            "mean_ms_per_step":  round(safe_mean("sma_infer_time_ms_per_step")  or 0, 4),
            "mean_rss_delta_KB": round(safe_mean("sma_infer_mem_KB")  or 0, 2),
        },
        # --- EWMA ---
        "ewma": {
            "mean_ms_per_step":  round(safe_mean("ewma_infer_time_ms_per_step") or 0, 4),
            "mean_rss_delta_KB": round(safe_mean("ewma_infer_mem_KB") or 0, 2),
        },
        # --- LAST ---
        "last": {
            "mean_ms_per_step":  round(safe_mean("last_infer_time_ms_per_step") or 0, 4),
            "mean_rss_delta_KB": round(safe_mean("last_infer_mem_KB") or 0, 2),
        },
    }


# ==============================================================================
#  Relative cost table
# ==============================================================================

def build_relative_table(summary):
    """Build a dict mapping method → {metric: value, ...} for the report table."""
    rows = []

    # Training time (Transformer only; others = 0 or fit-time)
    s = summary

    trans_train_s  = s["trans_train"]["mean_per_sig_s"]
    arima_fit_s    = s["arima"]["mean_fit_time_s"]
    sma_baseline   = s["sma"]["mean_ms_per_step"]

    # Relative inference vs SMA
    def rel_ms(ms):
        if sma_baseline and sma_baseline > 0:
            return round(ms / sma_baseline, 1)
        return None

    rows = [
        {
            "Method":           "Transformer (compact)",
            "Training (s/sig)": f"{trans_train_s:.3f} ± {s['trans_train']['std_per_sig_s']:.3f}",
            "Infer (ms/step)":  f"{s['trans_infer']['mean_ms_per_step']:.4f}",
            "Infer rel. SMA":   f"{rel_ms(s['trans_infer']['mean_ms_per_step'])}×",
            "Train mem (KB)":   f"{s['trans_train']['mean_rss_delta_KB']:.1f}",
            "Infer mem (KB)":   f"{s['trans_infer']['mean_rss_delta_KB']:.1f}",
            "# params":         str(s["n_params"]),
        },
        {
            "Method":           "ARIMA(1,1,1)",
            "Training (s/sig)": f"{arima_fit_s:.3f} (fit)",
            "Infer (ms/step)":  f"{s['arima']['mean_ms_per_step']:.3f}",
            "Infer rel. SMA":   f"{rel_ms(s['arima']['mean_ms_per_step'])}×",
            "Train mem (KB)":   f"{s['arima']['mean_fit_mem_KB']:.1f}",
            "Infer mem (KB)":   f"{s['arima']['mean_infer_mem_KB']:.1f}",
            "# params":         "N/A",
        },
        {
            "Method":           f"SMA(w={s['transformer_hp']['sma_w']})",
            "Training (s/sig)": "0 (none)",
            "Infer (ms/step)":  f"{s['sma']['mean_ms_per_step']:.4f}",
            "Infer rel. SMA":   "1.0× (baseline)",
            "Train mem (KB)":   "0",
            "Infer mem (KB)":   f"{s['sma']['mean_rss_delta_KB']:.1f}",
            "# params":         "N/A",
        },
        {
            "Method":           f"EWMA(α={s['transformer_hp']['ewma_alpha']})",
            "Training (s/sig)": "0 (none)",
            "Infer (ms/step)":  f"{s['ewma']['mean_ms_per_step']:.4f}",
            "Infer rel. SMA":   f"{rel_ms(s['ewma']['mean_ms_per_step'])}×",
            "Train mem (KB)":   "0",
            "Infer mem (KB)":   f"{s['ewma']['mean_rss_delta_KB']:.1f}",
            "# params":         "N/A",
        },
        {
            "Method":           "LAST",
            "Training (s/sig)": "0 (none)",
            "Infer (ms/step)":  f"{s['last']['mean_ms_per_step']:.4f}",
            "Infer rel. SMA":   f"{rel_ms(s['last']['mean_ms_per_step'])}×",
            "Train mem (KB)":   "0",
            "Infer mem (KB)":   f"{s['last']['mean_rss_delta_KB']:.1f}",
            "# params":         "N/A",
        },
    ]
    return rows


# ==============================================================================
#  Plotting
# ==============================================================================

def plot_efficiency(df, summary, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Transformer Efficiency vs Classical Detectors — {dataset_name}",
                 fontsize=13, fontweight="bold")

    # 1. Training time per signature
    ax = axes[0]
    methods_train = ["Transformer", "ARIMA (fit)", "SMA", "EWMA", "LAST"]
    train_times   = [
        summary["trans_train"]["mean_per_sig_s"],
        summary["arima"]["mean_fit_time_s"],
        0.0, 0.0, 0.0
    ]
    train_stds = [
        summary["trans_train"]["std_per_sig_s"],
        0.0, 0.0, 0.0, 0.0
    ]
    colors = ["#9467bd", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(methods_train, train_times, color=colors, alpha=0.85,
                  yerr=train_stds, capsize=5)
    ax.set_ylabel("Training time (s/sig)")
    ax.set_title("Training cost per signature")
    ax.set_yscale("symlog", linthresh=0.001)
    for bar, val in zip(bars, train_times):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f"{val:.3f}s", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 2. Inference time per step (ms)
    ax = axes[1]
    methods_inf = ["Transformer", "ARIMA", "SMA", "EWMA", "LAST"]
    inf_ms = [
        summary["trans_infer"]["mean_ms_per_step"],
        summary["arima"]["mean_ms_per_step"],
        summary["sma"]["mean_ms_per_step"],
        summary["ewma"]["mean_ms_per_step"],
        summary["last"]["mean_ms_per_step"],
    ]
    bars = ax.bar(methods_inf, inf_ms, color=colors, alpha=0.85)
    ax.set_ylabel("Inference time (ms/step)")
    ax.set_title("Per-step inference cost")
    ax.set_yscale("log")
    for bar, val in zip(bars, inf_ms):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 3. Peak memory (KB, training for Transformer, inference for classical)
    ax = axes[2]
    methods_mem = ["Trans\n(train)", "Trans\n(infer)", "ARIMA\n(infer)",
                   "SMA\n(infer)", "EWMA\n(infer)", "LAST\n(infer)"]
    mem_vals = [
        summary["trans_train"]["mean_rss_delta_KB"],
        summary["trans_infer"]["mean_rss_delta_KB"],
        summary["arima"]["mean_infer_mem_KB"],
        summary["sma"]["mean_rss_delta_KB"],
        summary["ewma"]["mean_rss_delta_KB"],
        summary["last"]["mean_rss_delta_KB"],
    ]
    mem_colors = ["#9467bd", "#c5b0d5", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(methods_mem, mem_vals, color=mem_colors, alpha=0.85)
    ax.set_ylabel("Approx. process RSS delta (KB/sig)")
    ax.set_title("Approx. process RSS delta (psutil, before/after)")
    for bar, val in zip(bars, mem_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"efficiency_{dataset_name.replace('-','_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_training_dist(df, dataset_name):
    """Distribution of per-signature Transformer training times."""
    vals = df["trans_train_time_s"].dropna().values
    if len(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=30, color="#9467bd", alpha=0.8, edgecolor="white")
    ax.axvline(vals.mean(), color="red", ls="--", lw=1.5,
               label=f"Mean = {vals.mean():.3f}s")
    ax.axvline(np.median(vals), color="orange", ls=":", lw=1.5,
               label=f"Median = {np.median(vals):.3f}s")
    ax.set_xlabel("Training time per signature (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Compact Transformer Training Time Distribution — {dataset_name}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"training_dist_{dataset_name.replace('-','_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_scatter_len_vs_time(df, dataset_name):
    """Scatter: series length vs training time."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["T_train"], df["trans_train_time_s"], alpha=0.4,
               s=20, color="#9467bd")
    ax.set_xlabel("Training-set size (T_train)")
    ax.set_ylabel("Transformer training time (s)")
    ax.set_title(f"Training Time vs Series Length — {dataset_name}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"train_vs_len_{dataset_name.replace('-','_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("  R1: Transformer Efficiency Measurement")
    print("="*70)

    all_df       = []
    all_summaries= {}

    for dataset_name, repo_name, ts_subfolder in DATASETS:
        ts_dir   = os.path.join(TS_BASE, ts_subfolder)
        max_sigs = MAX_SIGS_PER_DATASET.get(dataset_name)

        df, n_total, n_params = measure_dataset(
            dataset_name, repo_name, ts_dir, max_sigs)

        if df.empty:
            print(f"  No data for {dataset_name}. Skipping.")
            continue

        n_measured = len(df)
        hp         = DATASET_HP[dataset_name]
        summary    = aggregate(df, n_total, n_measured, dataset_name, hp)
        all_summaries[dataset_name] = summary
        all_df.append(df)

        # Print compact summary
        print(f"\n  --- {dataset_name} Summary ---")
        print(f"  Signatures measured : {n_measured} / {n_total}")
        print(f"  Transformer HP      : window={hp['window']}, d_model={hp['d_model']}")
        print(f"  # Parameters        : {n_params}")
        print(f"  Train time/sig      : {summary['trans_train']['mean_per_sig_s']:.3f} ± "
              f"{summary['trans_train']['std_per_sig_s']:.3f} s")
        print(f"  Train mem/sig       : {summary['trans_train']['mean_rss_delta_KB']:.1f} KB (approx. RSS delta)")
        print(f"  Infer time/step     : {summary['trans_infer']['mean_ms_per_step']:.4f} ms")
        print(f"  ARIMA fit time/sig  : {summary['arima']['mean_fit_time_s']:.3f} s")
        print(f"  ARIMA infer/step    : {summary['arima']['mean_ms_per_step']:.3f} ms")
        print(f"  SMA infer/step      : {summary['sma']['mean_ms_per_step']:.4f} ms")
        print(f"  Extrap total train  : {summary['trans_train']['extrap_total_s']:.1f} s "
              f"({summary['trans_train']['extrap_total_s']/3600:.2f} h)")

        # Relative table
        print(f"\n  Efficiency table ({dataset_name}):")
        tbl = build_relative_table(summary)
        for r in tbl:
            print(f"    {r['Method']:<30}  train={r['Training (s/sig)']:<20}  "
                  f"infer={r['Infer (ms/step)']:<12}  rel={r['Infer rel. SMA']:<15}  "
                  f"mem_train={r['Train mem (KB)']:<10}  params={r['# params']}")

        # Plots
        plot_efficiency(df, summary, dataset_name)
        plot_training_dist(df, dataset_name)
        plot_scatter_len_vs_time(df, dataset_name)

    # Save combined detail CSV
    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        csv_path = os.path.join(RESULTS_DIR, "transformer_efficiency_detail.csv")
        combined.to_csv(csv_path, index=False)
        print(f"\n  Detail CSV saved: {csv_path}")

    # Save summary JSON
    json_path = os.path.join(RESULTS_DIR, "transformer_efficiency_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"  Summary JSON saved: {json_path}")

    print("\n  Done — R1 Transformer Efficiency.")


if __name__ == "__main__":
    main()
