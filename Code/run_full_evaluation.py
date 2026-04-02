# ==============================================================================
# run_full_evaluation.py
# Full dual-dataset evaluation: all five methods on both firefox-android and
# mozilla-beta, using a fair three-way 60/20/20 chronological split with
# validation-set hyperparameter tuning and multi-seed Transformer.
#
# METHODS  (all share identical anomaly-score gating: Algorithm 3)
#   1. ARIMA(1,1,1)        — order held fixed from ablation study
#   2. LAST                — naive last-value predictor
#   3. SMA(w*)             — window w tuned on validation set
#   4. EWMA(alpha*)        — alpha tuned on validation set
#   5. Transformer(compact) — D_MODEL=16, 1 encoder layer, WINDOW=20,
#                             MSE+Adam; WINDOW and D_MODEL tuned on val set;
#                             5 independent seeds; mean +/- std reported
#
# SPLIT DESIGN  (identical for all methods)
#   Train 60%  — model fitting + delta_hist calibration
#   Val   20%  — SMA w, EWMA alpha, Transformer HP grid selection (F1)
#   Test  20%  — final metrics (NEVER used during tuning)
#
# OUTPUTS  (under results/<dataset>/)
#   detail.csv          — per-signature test-set results for all methods
#   summary.json        — aggregate metrics + tuned params
#   plots/              — val tuning curves, comparison bar chart, seed stability
#
# COMBINED OUTPUTS  (under results/)
#   combined_summary.json
#   combined_detail.csv
#   plots/combined_comparison.png
#
# Usage:
#   python Code/run_full_evaluation.py
# ==============================================================================

import os, sys, re, glob, json, math, time, warnings
warnings.filterwarnings("ignore")

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
    print("WARNING: PyTorch not found. Transformer will be skipped.")

# ==============================================================================
#  Paths
# ==============================================================================
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "Data")
TS_BASE    = os.path.join(DATA_DIR, "timeseries-data")
ALERTS_CSV = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS    = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Datasets to evaluate: (name, repo_name_in_alerts, ts_subfolder)
DATASETS = [
    ("firefox-android", "firefox-android", "firefox-android"),
    ("mozilla-beta",    "mozilla-beta",    "mozilla-beta"),
]

# ==============================================================================
#  Shared fixed parameters (Algorithm 3 anomaly-score gating)
# ==============================================================================
BETA              = 0.30
THRESHOLD_ANOMALY = 0.30
K_SIGMA           = 3.0
THR_WINDOW        = 30
MIN_THRESHOLD     = 1e-6
TOL               = 5       # ±5 sample tolerance for alert matching

ARIMA_ORDER       = (1, 1, 1)
ARIMA_MIN_HISTORY = 20
ARIMA_WINDOW      = 60

# Three-way split ratios
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20

# Tuning grids (applied to validation set)
SMA_WINDOWS   = [5, 10, 15, 20, 25, 30, 40, 50]
EWMA_ALPHAS   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Transformer HP grid (tuned on validation set)
TRANS_WINDOWS  = [10, 20, 30]          # sliding input window sizes to try
TRANS_DMODELS  = [8, 16, 32]           # embedding dimensions to try
TRANS_SEEDS    = [42, 123, 456, 789, 1024]

# Fixed Transformer training settings (not tuned)
TRANS_NHEAD        = 2     # always 2 (must divide D_MODEL; smallest D_MODEL=8 supports 2)
TRANS_NLAYERS      = 1
TRANS_DIM_FF_RATIO = 2     # DIM_FEEDFORWARD = D_MODEL * ratio
TRANS_DROPOUT      = 0.0
TRANS_LR           = 1e-2
TRANS_EPOCHS       = 20
TRANS_BATCH        = 256
TRANS_PATIENCE     = 6

# ==============================================================================
#  Utility helpers
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


def json_sanitize(obj):
    if obj is None: return None
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    try:
        if pd.isna(obj): return None
    except Exception: pass
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):  return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [json_sanitize(v) for v in obj]
    return obj


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
#  Evaluation metrics (interval-level precision, alert-level recall)
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
#  Classical detector: warmup + segment runner
# ==============================================================================

def arima_rolling_preds(arima_fit, seed_y, y_segment):
    params = arima_fit.params
    buf    = list(np.array(seed_y[-ARIMA_WINDOW:], dtype=float))
    preds  = []
    for val in y_segment:
        pr = None
        if len(buf) >= ARIMA_MIN_HISTORY:
            try:
                m   = StatsARIMA(np.array(buf, dtype=float), order=ARIMA_ORDER,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
                res = m.filter(params)
                pr  = safe_float(res.forecast(1)[0])
            except Exception:
                pass
        if pr is None:
            pr = safe_float(buf[-1])
        preds.append(pr)
        buf.append(float(val))
        if len(buf) > ARIMA_WINDOW: buf.pop(0)
    return np.array(preds)


def warmup(y_train, method, param, arima_fit=None):
    dh, sc = [], 0.0
    ewma   = float(y_train[0]) if len(y_train) > 0 else 0.0
    sma_w  = param if method == "sma" else 20
    alpha  = param if method == "ewma" else 0.30
    min_h  = sma_w if method == "sma" else (ARIMA_MIN_HISTORY if method == "arima" else 2)

    for t in range(len(y_train)):
        act = safe_float(y_train[t])
        if t < min_h or act is None:
            if method == "ewma" and t > 0 and act is not None:
                ewma = alpha * act + (1 - alpha) * ewma
            continue
        if method == "arima":
            if arima_fit is not None and t < len(arima_fit.fittedvalues):
                fv = arima_fit.fittedvalues
                pr = safe_float(fv.iloc[t] if hasattr(fv, "iloc") else fv[t])
            else:
                pr = safe_float(y_train[t - 1])
        elif method == "last":
            pr = safe_float(y_train[t - 1])
        elif method == "sma":
            pr = safe_float(np.mean(y_train[max(0, t - sma_w): t]))
        elif method == "ewma":
            pr   = ewma
            ewma = alpha * act + (1 - alpha) * ewma
        else:
            pr = None
        if pr is None: continue
        delta = abs(act - pr)
        is_a  = 1 if delta > dyn_thr(dh) else 0
        sc    = update_score(sc, is_a)
        dh.append(delta)

    return dh, sc, ewma


def run_segment(y_history, y_segment, method, param,
                init_dh, init_sc, init_ewma, arima_preds=None):
    dh, sc, ewma = list(init_dh), init_sc, init_ewma
    sma_w  = param if method == "sma" else 20
    alpha  = param if method == "ewma" else 0.30
    prev   = float(y_history[-1]) if len(y_history) > 0 else float(y_segment[0])
    decisions = []
    for i_t, val in enumerate(y_segment):
        act = safe_float(val)
        if act is None: decisions.append(0); continue
        if method == "arima":
            pr = (safe_float(arima_preds[i_t])
                  if arima_preds is not None and i_t < len(arima_preds)
                  else safe_float(y_segment[i_t - 1] if i_t > 0 else prev))
        elif method == "last":
            pr = safe_float(y_segment[i_t - 1] if i_t > 0 else prev)
        elif method == "sma":
            buf = np.concatenate([y_history, y_segment[:i_t]])[-sma_w:]
            pr  = safe_float(np.mean(buf)) if len(buf) > 0 else None
        elif method == "ewma":
            pr   = ewma
            ewma = alpha * act + (1 - alpha) * ewma
        else:
            pr = None
        if pr is None: decisions.append(0); continue
        delta = abs(act - pr)
        is_a  = 1 if delta > dyn_thr(dh) else 0
        sc    = update_score(sc, is_a)
        dh.append(delta)
        decisions.append(1 if sc >= THRESHOLD_ANOMALY else 0)
    return decisions, dh, sc, ewma


# ==============================================================================
#  Compact Transformer
# ==============================================================================

if TORCH_OK:
    class TSTransformer(nn.Module):
        def __init__(self, d_model, nhead, dim_ff, window):
            super().__init__()
            self.window   = window
            self.proj_in  = nn.Linear(1, d_model)
            enc_layer     = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=TRANS_DROPOUT, batch_first=True)
            self.encoder  = nn.TransformerEncoder(enc_layer,
                                                   num_layers=TRANS_NLAYERS)
            self.proj_out = nn.Linear(d_model, 1)

        def forward(self, x):          # x: (B, window, 1)
            h = self.proj_in(x)
            h = self.encoder(h)
            return self.proj_out(h[:, -1, :]).squeeze(-1)


    def train_transformer(y_train, window, d_model, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu")
        nhead  = min(TRANS_NHEAD, d_model)     # safety: nhead <= d_model
        dim_ff = d_model * TRANS_DIM_FF_RATIO

        model = TSTransformer(d_model, nhead, dim_ff, window).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=TRANS_LR)
        loss_fn = nn.MSELoss()

        if len(y_train) <= window + 1:
            return None

        y_t  = torch.tensor(y_train, dtype=torch.float32)
        # Normalise
        mu   = y_t.mean(); sig = y_t.std(); sig = sig if sig > 1e-8 else torch.tensor(1.0)
        y_n  = (y_t - mu) / sig

        # Build windows
        X = torch.stack([y_n[i: i + window] for i in range(len(y_n) - window)])
        Y = y_n[window:]
        X = X.unsqueeze(-1)

        ds   = torch.utils.data.TensorDataset(X, Y)
        dl   = torch.utils.data.DataLoader(ds, batch_size=TRANS_BATCH, shuffle=True)

        best_loss = float("inf"); patience_ct = 0; best_state = None
        for _ in range(TRANS_EPOCHS):
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
                if patience_ct >= TRANS_PATIENCE: break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return (model, mu.item(), sig.item(), window)


    def transformer_preds(model_tuple, y_series):
        """Return 1-step-ahead predictions as numpy array (same length as y_series)."""
        if model_tuple is None:
            return np.full(len(y_series), np.nan)
        model, mu, sig, window = model_tuple
        preds = [np.nan] * len(y_series)
        y_n   = (np.array(y_series, dtype=np.float32) - mu) / max(sig, 1e-8)
        model.eval()
        with torch.no_grad():
            for i in range(window, len(y_series)):
                buf  = torch.tensor(y_n[i - window: i], dtype=torch.float32)
                buf  = buf.unsqueeze(0).unsqueeze(-1)
                p    = model(buf).item() * max(sig, 1e-8) + mu
                preds[i] = p
        return np.array(preds)


    def run_segment_transformer(y_history, y_segment, model_tuple,
                                 init_dh, init_sc):
        """Run transformer predictor + Algorithm-3 gating."""
        dh, sc = list(init_dh), init_sc
        if model_tuple is None:
            return [0] * len(y_segment), dh, sc

        model, mu, sig, window = model_tuple
        # Build combined series for look-back
        combined = np.concatenate([y_history, y_segment])
        offset   = len(y_history)  # y_segment starts here

        combined_n = (combined.astype(np.float32) - mu) / max(sig, 1e-8)
        decisions  = []

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
        return decisions, dh, sc


    _HP_TUNE_SEED = 42   # single fixed seed for HP search (reproducible, fast)

    def transformer_val_f1(window, d_model, sigs_y, splits, alert_t_map):
        """Mean F1 on validation set using a single fixed seed for speed.
        The best (window, d_model) is then evaluated with all TRANS_SEEDS on test."""
        f1_list = []
        for sig, y in sigs_y:
            if sig not in splits: continue
            y_train, y_val, _, T_train, T_val, _ = splits[sig]
            if len(y_train) <= window + 1: continue
            mt = train_transformer(y_train, window, d_model, _HP_TUNE_SEED)
            if mt is None: continue
            dh0, sc0 = warmup_transformer(y_train, mt)
            dec_val, _, _ = run_segment_transformer(y_train, y_val, mt, dh0, sc0)
            m = seg_metrics(dec_val, T_train, alert_t_map.get(sig, []))
            f1_list.append(m["f1"])
        return float(np.mean(f1_list)) if f1_list else 0.0


    def warmup_transformer(y_train, model_tuple):
        """Run transformer on train set to build initial delta_hist and score."""
        dh, sc = [], 0.0
        if model_tuple is None:
            return dh, sc
        model, mu, sig, window = model_tuple
        y_n = (np.array(y_train, dtype=np.float32) - mu) / max(sig, 1e-8)
        model.eval()
        with torch.no_grad():
            for i in range(window, len(y_train)):
                buf   = torch.tensor(y_n[i - window: i],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                pr    = model(buf).item() * max(sig, 1e-8) + mu
                act   = safe_float(y_train[i])
                if act is None: continue
                delta = abs(act - pr)
                is_a  = 1 if delta > dyn_thr(dh) else 0
                sc    = update_score(sc, is_a)
                dh.append(delta)
        return dh, sc


# ==============================================================================
#  Data loading helpers
# ==============================================================================

def load_timeseries_for_dataset(ts_dir, alerts_df, repo_name):
    """
    Load and filter timeseries CSVs for one dataset.
    Returns list of (sig_str, y_array, alert_t_indices).
    """
    # Only use signature IDs that appear in alerts for this repository
    repo_alerts = alerts_df[alerts_df["alert_summary_repository"] == repo_name].copy()
    repo_alerts["sig_norm"] = repo_alerts["signature_id"].apply(normalize_sig)
    repo_alerts["push_timestamp"] = pd.to_datetime(
        repo_alerts["push_timestamp"], errors="coerce")
    repo_alerts = repo_alerts.dropna(subset=["push_timestamp"])

    alert_map = {}
    for sig, grp in repo_alerts.groupby("sig_norm"):
        alert_map[sig] = grp["push_timestamp"].tolist()

    # Find CSV files
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
            if len(y) < 30: continue   # minimum usable length

            # Map alert timestamps to t-index
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
        except Exception as e:
            pass

    # Sort by descending length (long series first — consistent with original)
    results.sort(key=lambda x: -len(x[1]))
    return results


# ==============================================================================
#  Per-dataset evaluation engine
# ==============================================================================

def evaluate_dataset(dataset_name, repo_name, ts_dir):
    """
    Run all methods on one dataset. Returns (detail_rows, summary_dict).
    """
    ds_results = os.path.join(RESULTS, dataset_name)
    ds_plots   = os.path.join(ds_results, "plots")
    os.makedirs(ds_plots, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}  (repo={repo_name})")
    print(f"{'='*70}")

    # ── Load data ──────────────────────────────────────────────────────────
    alerts_df = pd.read_csv(ALERTS_CSV)
    sigs_data = load_timeseries_for_dataset(ts_dir, alerts_df, repo_name)
    print(f"  Loaded {len(sigs_data)} valid signatures (len >= 30)")
    sigs_with_alerts = sum(1 for _, _, at in sigs_data if at)
    print(f"  Signatures with >= 1 mapped alert: {sigs_with_alerts}")

    if len(sigs_data) == 0:
        print("  ERROR: No data. Skipping.")
        return [], {}

    # ── Compute three-way splits + fit ARIMA ──────────────────────────────
    print("\n  Phase A: Computing splits and fitting ARIMA ...")
    splits     = {}
    arima_fits = {}
    for sig, y, at_list in sigs_data:
        T       = len(y)
        T_train = max(ARIMA_MIN_HISTORY + 2, int(round(TRAIN_RATIO * T)))
        T_val   = max(2, int(round(VAL_RATIO   * T)))
        T_test  = T - T_train - T_val
        if T_test < 2: continue
        splits[sig] = (y[:T_train],
                       y[T_train: T_train + T_val],
                       y[T_train + T_val:],
                       T_train, T_val, T_test)
        try:
            arima_fits[sig] = StatsARIMA(
                y[:T_train], order=ARIMA_ORDER,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit()
        except Exception:
            arima_fits[sig] = None

    alert_t_map = {sig: at for sig, _, at in sigs_data}
    valid_sigs  = [(sig, y, at) for sig, y, at in sigs_data if sig in splits]
    sigs_y      = [(sig, y) for sig, y, _ in valid_sigs]
    print(f"  Valid signatures (T_test >= 2): {len(splits)}")
    print(f"  ARIMA fits ok: {sum(1 for v in arima_fits.values() if v is not None)}/{len(splits)}")

    # ── Phase B: Validation tuning - classical methods ─────────────────────
    print("\n  Phase B: Validation tuning (classical) ...")

    def mean_val_f1_classical(method, param):
        f1s = []
        for sig, y in sigs_y:
            if sig not in splits: continue
            y_train, y_val, _, T_train, _, _ = splits[sig]
            if method == "sma" and len(y_train) < param: continue
            dh0, sc0, ew0 = warmup(y_train, method, param, arima_fits.get(sig))
            dec, _, _, _  = run_segment(y_train, y_val, method, param,
                                        dh0, sc0, ew0)
            f1s.append(seg_metrics(dec, T_train, alert_t_map.get(sig, []))["f1"])
        return float(np.mean(f1s)) if f1s else 0.0

    sma_vf1  = {w: mean_val_f1_classical("sma",  w) for w in SMA_WINDOWS}
    ewma_vf1 = {a: mean_val_f1_classical("ewma", a) for a in EWMA_ALPHAS}
    best_sma_w         = max(sma_vf1,  key=sma_vf1.get)
    best_ewma_alpha    = max(ewma_vf1, key=ewma_vf1.get)
    print(f"  Best SMA  w     = {best_sma_w}   (val F1={sma_vf1[best_sma_w]:.4f})")
    print(f"  Best EWMA alpha = {best_ewma_alpha}  (val F1={ewma_vf1[best_ewma_alpha]:.4f})")

    # SMA tuning plot
    _plot_tuning(list(sma_vf1.keys()), list(sma_vf1.values()),
                 best_sma_w, "SMA Window w", "Mean Validation F1",
                 f"SMA Window Tuning — {dataset_name}",
                 os.path.join(ds_plots, "val_tuning_sma.png"))
    # EWMA tuning plot
    _plot_tuning(list(ewma_vf1.keys()), list(ewma_vf1.values()),
                 best_ewma_alpha, "EWMA alpha", "Mean Validation F1",
                 f"EWMA Alpha Tuning — {dataset_name}",
                 os.path.join(ds_plots, "val_tuning_ewma.png"))

    # ── Phase C: Transformer HP tuning ────────────────────────────────────
    best_trans_window  = TRANS_WINDOWS[1]   # default fallback = 20
    best_trans_dmodel  = TRANS_DMODELS[1]   # default fallback = 16
    trans_hp_results   = {}

    if TORCH_OK:
        print("\n  Phase C: Transformer HP tuning on validation set ...")
        print(f"  Grid: windows={TRANS_WINDOWS} x d_models={TRANS_DMODELS} "
              f"x {len(TRANS_SEEDS)} seeds")
        for tw in TRANS_WINDOWS:
            for td in TRANS_DMODELS:
                key = (tw, td)
                vf1 = transformer_val_f1(tw, td, sigs_y, splits, alert_t_map)
                trans_hp_results[key] = vf1
                print(f"    window={tw}, d_model={td}: val F1 = {vf1:.4f}")
        best_key           = max(trans_hp_results, key=trans_hp_results.get)
        best_trans_window  = best_key[0]
        best_trans_dmodel  = best_key[1]
        print(f"  Best Transformer: window={best_trans_window}, "
              f"d_model={best_trans_dmodel} "
              f"(val F1={trans_hp_results[best_key]:.4f})")
        # HP heatmap
        _plot_hp_heatmap(trans_hp_results, TRANS_WINDOWS, TRANS_DMODELS,
                         dataset_name, ds_plots)
    else:
        print("  Phase C: Skipping Transformer (PyTorch unavailable).")

    # ── Phase D: Test-set evaluation ──────────────────────────────────────
    print("\n  Phase D: Test-set evaluation ...")

    detail_rows = []

    # Classical methods
    classical = [
        ("ARIMA(1,1,1)", "arima",  None),
        ("LAST",         "last",   None),
        (f"SMA(w={best_sma_w})",           "sma",  best_sma_w),
        (f"EWMA(a={best_ewma_alpha})",      "ewma", best_ewma_alpha),
    ]

    for label, method, param in classical:
        print(f"    {label} ...", end="", flush=True)
        t0 = time.time()
        for sig, y in sigs_y:
            if sig not in splits: continue
            y_train, y_val, y_test, T_train, T_val, T_test = splits[sig]
            eff_param = param if param is not None else (20 if method == "sma" else 0.30)
            dh0, sc0, ew0 = warmup(y_train, method, eff_param, arima_fits.get(sig))
            # Run val to advance state correctly (don't re-tune, just propagate state)
            arima_preds_val = None
            if method == "arima" and arima_fits.get(sig):
                arima_preds_val = arima_rolling_preds(arima_fits[sig], y_train, y_val)
            dec_val, dh1, sc1, ew1 = run_segment(y_train, y_val, method, eff_param,
                                                   dh0, sc0, ew0, arima_preds_val)
            # Test
            arima_preds_test = None
            if method == "arima" and arima_fits.get(sig):
                arima_preds_test = arima_rolling_preds(
                    arima_fits[sig], np.concatenate([y_train, y_val]), y_test)
            dec_test, _, _, _ = run_segment(
                np.concatenate([y_train, y_val]), y_test,
                method, eff_param, dh1, sc1, ew1, arima_preds_test)
            m = seg_metrics(dec_test, T_train + T_val, alert_t_map.get(sig, []))
            detail_rows.append(dict(
                dataset=dataset_name, method=label, signature_id=sig,
                T_total=len(y), T_train=T_train, T_val=T_val, T_test=T_test,
                param=str(eff_param),
                seed=None,
                **{f"test_{k}": v for k, v in m.items()}
            ))
        print(f" {time.time()-t0:.0f}s")

    # Transformer (5 seeds)
    if TORCH_OK:
        trans_label = f"Transformer(w={best_trans_window},d={best_trans_dmodel})"
        print(f"    {trans_label} (5 seeds) ...", flush=True)
        t0 = time.time()
        for seed in TRANS_SEEDS:
            print(f"      seed={seed} ...", end="", flush=True)
            t1 = time.time()
            for sig, y in sigs_y:
                if sig not in splits: continue
                y_train, y_val, y_test, T_train, T_val, T_test = splits[sig]
                if len(y_train) <= best_trans_window + 1: continue
                mt = train_transformer(y_train, best_trans_window,
                                       best_trans_dmodel, seed)
                if mt is None: continue
                dh0, sc0 = warmup_transformer(y_train, mt)
                # advance state through val
                dec_val, dh1, sc1 = run_segment_transformer(y_train, y_val, mt, dh0, sc0)
                dec_test, _, _    = run_segment_transformer(
                    np.concatenate([y_train, y_val]), y_test, mt, dh1, sc1)
                m = seg_metrics(dec_test, T_train + T_val, alert_t_map.get(sig, []))
                detail_rows.append(dict(
                    dataset=dataset_name, method=trans_label, signature_id=sig,
                    T_total=len(y), T_train=T_train, T_val=T_val, T_test=T_test,
                    param=f"w={best_trans_window},d={best_trans_dmodel}",
                    seed=seed,
                    **{f"test_{k}": v for k, v in m.items()}
                ))
            print(f" {time.time()-t1:.0f}s")
        print(f"    Total Transformer time: {time.time()-t0:.0f}s")

    # ── Phase E: Aggregate summary ─────────────────────────────────────────
    print("\n  Phase E: Aggregating results ...")
    summary = _aggregate_summary(detail_rows, dataset_name,
                                 best_sma_w, best_ewma_alpha,
                                 best_trans_window if TORCH_OK else None,
                                 best_trans_dmodel if TORCH_OK else None,
                                 sma_vf1, ewma_vf1, trans_hp_results)

    # Save per-dataset files
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(ds_results, "detail.csv"), index=False)
    with open(os.path.join(ds_results, "summary.json"), "w") as f:
        json.dump(json_sanitize(summary), f, indent=2)
    print(f"  Saved: {ds_results}/detail.csv + summary.json")

    # Comparison plot
    _plot_comparison(summary, dataset_name, ds_plots)

    # Seed stability plot (Transformer only)
    if TORCH_OK:
        _plot_seed_stability(detail_rows, dataset_name, ds_plots)

    return detail_rows, summary


# ==============================================================================
#  Aggregation helpers
# ==============================================================================

def _aggregate_summary(detail_rows, dataset_name,
                        best_sma_w, best_ewma_alpha,
                        best_trans_window, best_trans_dmodel,
                        sma_vf1, ewma_vf1, trans_hp_results):
    df = pd.DataFrame(detail_rows)
    methods_summary = {}

    for method_label, grp in df.groupby("method"):
        is_transformer = "Transformer" in method_label
        if is_transformer:
            # Average across seeds per signature first, then aggregate
            per_sig = grp.groupby("signature_id").agg({
                "test_precision": "mean", "test_recall": "mean",
                "test_f1": "mean", "test_far": "mean",
                "test_storage_reduc": "mean", "test_detected": "mean",
                "test_has_alert": "first",
            }).reset_index()
            # Seed-level F1 means (for std calculation)
            seed_means = grp.groupby("seed")["test_f1"].mean().values
            std_f1     = float(np.std(seed_means, ddof=0)) if len(seed_means) > 1 else 0.0
            std_prec   = float(np.std(grp.groupby("seed")["test_precision"].mean().values, ddof=0))
            std_rec    = float(np.std(grp.groupby("seed")["test_recall"].mean().values, ddof=0))
            std_far    = float(np.std(grp.groupby("seed")["test_far"].mean().values, ddof=0))
            n_seeds    = int(grp["seed"].nunique())
            per_seed   = [{
                "seed": int(s),
                "mean_f1": round(float(sgrp["test_f1"].mean()), 4),
                "mean_precision": round(float(sgrp["test_precision"].mean()), 4),
                "mean_recall": round(float(sgrp["test_recall"].mean()), 4),
            } for s, sgrp in grp.groupby("seed")]
        else:
            per_sig    = grp
            std_f1 = std_prec = std_rec = std_far = 0.0
            n_seeds = 1; per_seed = []

        alerted = per_sig[per_sig["test_has_alert"] == True] \
                  if is_transformer else per_sig[per_sig["test_has_alert"] == True]
        det_rate = (alerted["test_detected"].mean()
                    if len(alerted) > 0 else 0.0) * 100

        methods_summary[method_label] = {
            "n_signatures":    len(per_sig),
            "n_with_alerts":   int((per_sig["test_has_alert"] == True).sum()),
            "mean_precision":  round(float(per_sig["test_precision"].mean()), 4),
            "std_precision":   round(std_prec, 4),
            "mean_recall":     round(float(per_sig["test_recall"].mean()), 4),
            "std_recall":      round(std_rec, 4),
            "mean_f1":         round(float(per_sig["test_f1"].mean()), 4),
            "std_f1":          round(std_f1, 4),
            "mean_far":        round(float(per_sig["test_far"].mean()), 6),
            "std_far":         round(std_far, 6),
            "mean_storage_reduc": round(float(per_sig["test_storage_reduc"].mean()), 4),
            "detection_rate_pct": round(det_rate, 2),
            "n_seeds":         n_seeds,
            "per_seed":        per_seed,
        }

    return {
        "dataset": dataset_name,
        "tuned_params": {
            "sma_window":    best_sma_w,
            "ewma_alpha":    best_ewma_alpha,
            "transformer_window": best_trans_window,
            "transformer_dmodel": best_trans_dmodel,
        },
        "val_tuning": {
            "sma":  {str(k): round(v, 4) for k, v in sma_vf1.items()},
            "ewma": {str(k): round(v, 4) for k, v in ewma_vf1.items()},
            "transformer": {str(k): round(v, 4) for k, v in trans_hp_results.items()},
        },
        "methods": methods_summary,
    }


# ==============================================================================
#  Plotting helpers
# ==============================================================================

def _plot_tuning(xs, f1s, best, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, f1s, "o-", color="#1f77b4", lw=2, markersize=7)
    ax.axvline(best, color="#d62728", ls="--", lw=1.5,
               label=f"Best = {best}  (F1={max(f1s):.4f})")
    ax.fill_between(xs, f1s, min(f1s) * 0.98, alpha=0.12, color="#1f77b4")
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_hp_heatmap(results, windows, dmodels, dataset_name, plots_dir):
    data = np.zeros((len(dmodels), len(windows)))
    for i, td in enumerate(dmodels):
        for j, tw in enumerate(windows):
            data[i, j] = results.get((tw, td), 0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlGn")
    ax.set_xticks(range(len(windows))); ax.set_xticklabels([str(w) for w in windows])
    ax.set_yticks(range(len(dmodels))); ax.set_yticklabels([str(d) for d in dmodels])
    ax.set_xlabel("Window size"); ax.set_ylabel("D_model")
    ax.set_title(f"Transformer HP Tuning (val F1) — {dataset_name}")
    for i in range(len(dmodels)):
        for j in range(len(windows)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_tuning_transformer.png"), dpi=150)
    plt.close(fig)


def _plot_comparison(summary, dataset_name, plots_dir):
    methods = list(summary["methods"].keys())
    metrics = ["mean_precision", "mean_recall", "mean_f1",
               "mean_far", "detection_rate_pct", "mean_storage_reduc"]
    labels  = ["Precision", "Recall", "F1", "FAR", "Det. Rate %", "Storage Red."]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]
    for ax, metric, label in zip(axes.flat, metrics, labels):
        vals = [summary["methods"][m].get(metric, 0) for m in methods]
        stds = [summary["methods"][m].get(f"std_{metric.replace('mean_','')}", 0) for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors[:len(methods)],
                      yerr=stds, capsize=4, alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10); ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"Method Comparison — {dataset_name} (3-way 60/20/20 split)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "comparison.png"), dpi=150)
    plt.close(fig)


def _plot_seed_stability(detail_rows, dataset_name, plots_dir):
    df = pd.DataFrame(detail_rows)
    trans_rows = df[df["method"].str.contains("Transformer") & df["seed"].notna()]
    if trans_rows.empty: return
    seed_f1 = trans_rows.groupby("seed")["test_f1"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(seed_f1["seed"].astype(str), seed_f1["test_f1"],
           color="#9467bd", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(seed_f1["test_f1"].mean(), color="red", ls="--", lw=1.5,
               label=f"Mean F1 = {seed_f1['test_f1'].mean():.4f}")
    ax.set_xlabel("Seed"); ax.set_ylabel("Mean F1 across signatures")
    ax.set_title(f"Transformer Seed Stability — {dataset_name}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "transformer_seed_stability.png"), dpi=150)
    plt.close(fig)


# ==============================================================================
#  Combined cross-dataset report
# ==============================================================================

def build_combined_report(all_details, all_summaries):
    print(f"\n{'='*70}")
    print("  Building combined cross-dataset report ...")
    print(f"{'='*70}")

    # Combined detail CSV
    combined_df = pd.DataFrame([row for rows in all_details for row in rows])
    combined_df.to_csv(os.path.join(RESULTS, "combined_detail.csv"), index=False)

    # Combined summary JSON
    combined_summary = {
        "datasets": {s["dataset"]: s for s in all_summaries},
    }

    # Cross-dataset aggregate: for each method, average across datasets
    all_methods = set()
    for s in all_summaries:
        all_methods.update(s["methods"].keys())

    cross = {}
    for method in sorted(all_methods):
        vals = [s["methods"][method] for s in all_summaries if method in s["methods"]]
        if not vals: continue
        cross[method] = {
            "datasets_present": [s["dataset"] for s in all_summaries if method in s["methods"]],
            "pooled_mean_precision":  round(np.mean([v["mean_precision"]  for v in vals]), 4),
            "pooled_mean_recall":     round(np.mean([v["mean_recall"]     for v in vals]), 4),
            "pooled_mean_f1":         round(np.mean([v["mean_f1"]         for v in vals]), 4),
            "pooled_mean_far":        round(np.mean([v["mean_far"]        for v in vals]), 6),
            "pooled_mean_storage_reduc": round(np.mean([v["mean_storage_reduc"] for v in vals]), 4),
            "pooled_detection_rate_pct": round(np.mean([v["detection_rate_pct"] for v in vals]), 2),
            "pooled_std_f1": round(float(np.std([v["mean_f1"] for v in vals], ddof=0)), 4),
        }
    combined_summary["cross_dataset_pooled"] = cross

    with open(os.path.join(RESULTS, "combined_summary.json"), "w") as f:
        json.dump(json_sanitize(combined_summary), f, indent=2)
    print(f"  Saved: results/combined_detail.csv + combined_summary.json")

    # Cross-dataset comparison plot
    _plot_cross_dataset(combined_summary, all_summaries)


def _plot_cross_dataset(combined_summary, all_summaries):
    cross = combined_summary["cross_dataset_pooled"]
    methods = list(cross.keys())
    datasets = [s["dataset"] for s in all_summaries]
    metrics  = [("mean_f1", "F1"), ("mean_precision", "Precision"),
                ("mean_recall", "Recall"), ("detection_rate_pct", "Det. Rate %"),
                ("mean_far", "FAR"), ("mean_storage_reduc", "Storage Red.")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors_ds = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    x = np.arange(len(methods))
    w = 0.35

    for ax, (metric, label) in zip(axes.flat, metrics):
        for i_ds, s in enumerate(all_summaries):
            vals = [s["methods"].get(m, {}).get(metric, 0) for m in methods]
            offset = (i_ds - (len(all_summaries) - 1) / 2) * w
            ax.bar(x + offset, vals, w * 0.9, label=s["dataset"],
                   color=colors_ds[i_ds], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Cross-Dataset Method Comparison  (60/20/20 three-way split)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "plots", "combined_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: results/plots/combined_comparison.png")


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("run_full_evaluation.py — Dual-Dataset Five-Method Evaluation")
    print("=" * 70)
    print(f"Datasets  : {[d[0] for d in DATASETS]}")
    print(f"Split     : train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test=20%")
    print(f"Transformer seeds : {TRANS_SEEDS}")
    print(f"Transformer HP grid: windows={TRANS_WINDOWS}, d_models={TRANS_DMODELS}")
    print()

    os.makedirs(os.path.join(RESULTS, "plots"), exist_ok=True)

    t_total = time.time()
    all_details   = []
    all_summaries = []

    for ds_name, repo_name, ts_subfolder in DATASETS:
        ts_dir = os.path.join(TS_BASE, ts_subfolder)
        if not os.path.isdir(ts_dir):
            print(f"WARNING: Timeseries directory not found: {ts_dir}. Skipping.")
            continue
        rows, summary = evaluate_dataset(ds_name, repo_name, ts_dir)
        if rows:
            all_details.append(rows)
            all_summaries.append(summary)

    if len(all_summaries) >= 1:
        build_combined_report(all_details, all_summaries)

    elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("Done.")
