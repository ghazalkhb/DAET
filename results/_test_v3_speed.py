# -*- coding: utf-8 -*-
"""
Quick test of run_transformer_v3 components on a single signature.
Does NOT import the module (avoids triggering main block).
"""
import os, sys, time, math
sys.path.insert(0, r'd:\Naser\2\Code')
os.chdir(r'd:\Naser\2')

import numpy as np
import torch
import torch.nn as nn

# Inline the needed constants and classes for isolated testing
WINDOW=30; D_MODEL=32; N_HEAD=4; N_LAYERS=2; DIM_FEEDFORWARD=128
DROPOUT=0.1; LR=5e-4; WEIGHT_DECAY=1e-4; HUBER_DELTA=1.0
EPOCHS=60; BATCH_SIZE=128; PATIENCE=12
BETA=0.30; ALPHA_BLEND=0.60; MIN_THRESHOLD=1e-6; K_SIGMA=3.0
ALPHA_MU=0.15; ALPHA_VAR=0.10
ANOM_HEAD_HIDDEN=16; ANOM_EPOCHS=15; ANOM_LR=5e-4; ANOM_PATIENCE=5
GRAC_HIDDEN=16; GRAC_EPOCHS=20; GRAC_LR=5e-4; GRAC_PATIENCE=5
GRAC_PSEUDO_SHARP=8.0; GRAC_EMA_ALPHA=0.40; MIN_GRAC_TRAIN=20
MIN_ANOM_TRAIN=30; THETA_PERCENTILE=90

torch.manual_seed(42); np.random.seed(42)

# Load first (longest) signature
VECTORS_DIR = r'd:\Naser\2\results\vectors'
files = sorted([f for f in os.listdir(VECTORS_DIR) if f.endswith('.npy')],
               key=lambda x: -len(np.load(os.path.join(VECTORS_DIR, x))))
y = np.load(os.path.join(VECTORS_DIR, files[0])).astype(float)
T = len(y); T_train = max(WINDOW+5, int(0.70*T)); y_train = y[:T_train]; y_test = y[T_train:]
print(f"Signature: {files[0]}, T={T}, T_train={T_train}, T_test={len(y_test)}")

# Minimal model definition (just to test GRAC and anomaly head shapes)
class TSTransformerV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1, D_MODEL)
        pos = torch.arange(WINDOW).unsqueeze(1).float()
        dim = torch.arange(0, D_MODEL, 2).float()
        div = torch.exp(-dim * math.log(10000.0) / D_MODEL)
        pe  = torch.zeros(WINDOW, D_MODEL)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))
        enc_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD,
            dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, batch_first=True, norm_first=True)
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.output_proj = nn.Linear(D_MODEL, 1)
        self.anom_head   = nn.Sequential(nn.Linear(D_MODEL, ANOM_HEAD_HIDDEN), nn.ReLU(), nn.Linear(ANOM_HEAD_HIDDEN, 1))
    def forward(self, x):
        h = self.input_proj(x) + self.pos_enc
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.output_proj(last).squeeze(-1), self.anom_head(last).squeeze(-1)

class GRAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(input_size=2, hidden_size=GRAC_HIDDEN, batch_first=True)
        self.proj = nn.Linear(GRAC_HIDDEN, 1)
    def forward_seq(self, seq, h0=None):
        out, h_n = self.gru(seq, h0)
        return self.proj(out.squeeze(0))
    def step(self, score_t, delta_norm_t, h):
        inp = torch.tensor([[[score_t, delta_norm_t]]], dtype=torch.float32)
        out, new_h = self.gru(inp, h)
        return torch.sigmoid(self.proj(out.squeeze(0).squeeze(0))).item(), new_h

print("\n--- Testing Phase 1 (Prediction training, 60 epochs) ---")
X_raw = np.lib.stride_tricks.sliding_window_view(y_train.astype(np.float64), WINDOW)[:-1].astype(np.float64)
mu    = X_raw.mean(axis=1); sigma = X_raw.std(axis=1) + 1e-8
X_norm = ((X_raw - mu[:,None]) / sigma[:,None]).astype(np.float32)
y_raw  = y_train[WINDOW:]
mu2    = X_raw.mean(axis=1); sigma2 = X_raw.std(axis=1) + 1e-8
y_norm = ((y_raw - mu2) / sigma2).astype(np.float32)
X_t = torch.tensor(X_norm[:, :, np.newaxis]); y_t = torch.tensor(y_norm)
model = TSTransformerV3()
pred_params = list(model.input_proj.parameters()) + list(model.encoder.parameters()) + list(model.output_proj.parameters())
opt = torch.optim.AdamW(pred_params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR/100)
crit = nn.HuberLoss(delta=HUBER_DELTA)
n = len(X_t); best_loss = 1e9; best_state = None; no_improve = 0
t0 = time.perf_counter()
model.train()
for ep in range(EPOCHS):
    perm = torch.randperm(n); eloss = 0.0; steps = 0
    for st in range(0, n, BATCH_SIZE):
        idx = perm[st:st+BATCH_SIZE]
        opt.zero_grad(); pred, _ = model(X_t[idx]); loss = crit(pred, y_t[idx])
        loss.backward(); nn.utils.clip_grad_norm_(pred_params, 1.0); opt.step()
        eloss += loss.item(); steps += 1
    scheduler.step()
    avg = eloss / max(steps,1)
    if avg < best_loss - 1e-6: best_loss = avg; best_state = {k:v.clone() for k,v in model.state_dict().items()}; no_improve=0
    else: no_improve+=1
    if no_improve >= PATIENCE: print(f"  Early stop at epoch {ep+1}"); break
t1 = time.perf_counter()
print(f"Phase 1 time: {t1-t0:.1f}s | best_loss={best_loss:.6f}")
if best_state: model.load_state_dict(best_state)
model.eval()

print("\n--- Testing Phase 2 (Anomaly head training) ---")
preds_all = []; anom_all = []
with torch.no_grad():
    for st in range(0, n, 512):
        xmu  = X_raw[st:st+512].mean(axis=1); xsig = X_raw[st:st+512].std(axis=1)+1e-8
        xn   = ((X_raw[st:st+512] - xmu[:,None])/xsig[:,None]).astype(np.float32)
        p, a = model(torch.tensor(xn[:,:,np.newaxis]))
        preds_all.append((p.numpy() * xsig + xmu)); anom_all.append(a.numpy())
preds_all = np.concatenate(preds_all); anom_all  = np.concatenate(anom_all)
residuals = np.abs(y_raw - preds_all)
delta_sigma = np.std(residuals[np.isfinite(residuals)]) + 1e-8
print(f"  delta_sigma={delta_sigma:.4f}")
z_scores = residuals / delta_sigma; z_scores = np.where(np.isfinite(z_scores), z_scores, 0.0)
labels_p2 = (1.0/(1.0+np.exp(-(z_scores-1.5)*2.0))).astype(np.float32)
contexts = []
with torch.no_grad():
    for st in range(0, n, 512):
        xmu  = X_raw[st:st+512].mean(axis=1); xsig = X_raw[st:st+512].std(axis=1)+1e-8
        xn   = ((X_raw[st:st+512] - xmu[:,None])/xsig[:,None]).astype(np.float32)
        xt   = torch.tensor(xn[:,:,np.newaxis])
        h    = model.input_proj(xt) + model.pos_enc; h = model.encoder(h)
        contexts.append(h[:,-1,:])
contexts = torch.cat(contexts, 0); lbl_p2 = torch.tensor(labels_p2).unsqueeze(1)
opt2 = torch.optim.Adam(model.anom_head.parameters(), lr=ANOM_LR)
crit2 = nn.BCEWithLogitsLoss()
model.train(); best2 = 1e9; pat2 = 0
t0 = time.perf_counter()
for ep in range(ANOM_EPOCHS):
    perm = torch.randperm(n); eloss = 0.0; steps = 0
    for st in range(0, n, BATCH_SIZE):
        idx = perm[st:st+BATCH_SIZE]; opt2.zero_grad()
        lo = model.anom_head(contexts[idx]); loss = crit2(lo, lbl_p2[idx])
        loss.backward(); opt2.step(); eloss += loss.item(); steps += 1
    avg = eloss / max(steps,1)
    if avg < best2 - 1e-6: best2 = avg; pat2 = 0
    else: pat2 += 1
    if pat2 >= ANOM_PATIENCE: print(f"  Early stop at epoch {ep+1}"); break
t1 = time.perf_counter()
print(f"Phase 2 time: {t1-t0:.1f}s")
model.eval()

print("\n--- Testing Phase 3 (GRAC training, vectorised) ---")
# Compute training scores
adt_mu = 0.0; adt_var = 0.0; adt_n = 0; score_tr = 0.0
sc_list=[]; dn_list=[]
for i in range(min(len(y_raw), n)):
    d = abs(float(y_raw[i]) - float(preds_all[i]))
    if not math.isfinite(d): continue
    if adt_n == 0: thr = 0.0
    else: thr = max(MIN_THRESHOLD, adt_mu + K_SIGMA * math.sqrt(adt_var + 1e-12))
    is_hard = 1.0 if d > thr else 0.0
    is_soft = float(1.0/(1.0+math.exp(-float(anom_all[i])))) if i < len(anom_all) else is_hard
    is_a = ALPHA_BLEND * is_soft + (1-ALPHA_BLEND) * is_hard
    score_tr = BETA * is_a + (1-BETA) * score_tr
    diff = d - adt_mu
    adt_mu  = (1-ALPHA_MU)*adt_mu  + ALPHA_MU*d
    adt_var = (1-ALPHA_VAR)*adt_var + ALPHA_VAR*diff*diff
    adt_n += 1
    sc_list.append(score_tr); dn_list.append(d / delta_sigma)
theta_sig = float(np.clip(np.percentile(sc_list, THETA_PERCENTILE), 0.10, 0.80))
print(f"  theta_sig={theta_sig:.4f}, seq_len={len(sc_list)}")
sc_arr = np.array(sc_list, dtype=np.float32); dn_arr = np.array(dn_list, dtype=np.float32)
raw = 1.0/(1.0+np.exp(-(sc_arr-theta_sig)*GRAC_PSEUDO_SHARP))
labels_p3 = np.zeros(len(raw), dtype=np.float32); ema = 0.0
for t in range(len(raw)):
    ema = GRAC_EMA_ALPHA * float(raw[t]) + (1-GRAC_EMA_ALPHA) * ema; labels_p3[t] = ema
seq = torch.stack([torch.tensor(sc_arr), torch.tensor(dn_arr)], dim=1).unsqueeze(0)
lbl_p3 = torch.tensor(labels_p3, dtype=torch.float32)
grac = GRAC(); opt3 = torch.optim.Adam(grac.parameters(), lr=GRAC_LR)
crit3 = nn.BCEWithLogitsLoss(); grac.train(); best3=1e9; pat3=0
t0 = time.perf_counter()
for ep in range(GRAC_EPOCHS):
    opt3.zero_grad()
    logits = grac.forward_seq(seq); loss = crit3(logits, lbl_p3.unsqueeze(1))
    loss.backward(); nn.utils.clip_grad_norm_(grac.parameters(), 1.0); opt3.step()
    lv = loss.item()
    if lv < best3 - 1e-6: best3=lv; pat3=0
    else: pat3+=1
    if pat3 >= GRAC_PATIENCE: print(f"  GRAC early stop epoch {ep+1}"); break
    if (ep+1) % 5 == 0: print(f"  GRAC epoch {ep+1}/{GRAC_EPOCHS} loss={lv:.4f} {time.perf_counter()-t0:.1f}s")
t1 = time.perf_counter()
print(f"Phase 3 (GRAC) time: {t1-t0:.1f}s for {len(sc_list)} steps x {GRAC_EPOCHS} epochs")

print(f"\n--- Inference test ---")
h_grac = torch.zeros(1, 1, GRAC_HIDDEN); grac.eval()
t0 = time.perf_counter()
decisions = []
for i in range(min(10, len(y_test))):
    g_prob, h_grac = grac.step(0.2, 0.5, h_grac)
    decisions.append(int(g_prob >= 0.5))
    print(f"  t={i}: gate_prob={g_prob:.3f} decision={decisions[-1]}")
print(f"\nEstimated total time for this sig (Phase1+Phase2+Phase3): ~{(t1-t0)+0:.0f}s for Phase3 alone")
