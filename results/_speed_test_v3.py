"""Speed test for V3 components: Phase 1, Phase 2, Phase 3, test loop."""
import os, sys, time
os.chdir(r'd:\Naser\2')
sys.path.insert(0, r'd:\Naser\2\Code')

import numpy as np, torch, torch.nn as nn

VECTORS_DIR = r'd:\Naser\2\results\vectors'
files = sorted([f for f in os.listdir(VECTORS_DIR) if f.endswith('.npy')],
               key=lambda x: -len(np.load(os.path.join(VECTORS_DIR, x))))

# Test with first (longest) signature
y = np.load(os.path.join(VECTORS_DIR, files[0])).astype(float)
print(f"Signature: {files[0]}, T={len(y)}")

WINDOW=30; TRAIN_RATIO=0.70
T       = len(y)
T_train = max(WINDOW + 5, int(TRAIN_RATIO * T))
y_train = y[:T_train]
y_test  = y[T_train:]
print(f"T_train={T_train}, T_test={len(y_test)}, seq={T_train-WINDOW}")

from run_transformer_v3 import (train_prediction_phase, train_anomaly_head_phase,
                                 batch_predict_v3, GRAC, GRAC_HIDDEN, GRAC_EPOCHS,
                                 GRAC_LR, GRAC_PATIENCE, GRAC_PSEUDO_SHARP, GRAC_EMA_ALPHA,
                                 MIN_GRAC_TRAIN, AdaptiveDeltaThreshold,
                                 BETA, ALPHA_BLEND)
import torch.nn as nn

print("\n--- Phase 1: Prediction training ---")
t0 = time.perf_counter()
model = train_prediction_phase(y_train)
t1 = time.perf_counter()
print(f"Phase 1: {t1-t0:.1f}s")

print("\n--- Phase 2: Anomaly head fine-tuning ---")
X_tr = np.lib.stride_tricks.sliding_window_view(y_train, WINDOW)[:-1]
tr_preds, _ = batch_predict_v3(model, X_tr)
y_tr_act = y_train[WINDOW:]
tr_deltas = np.abs(y_tr_act - tr_preds)
delta_sigma = float(np.std(tr_deltas[np.isfinite(tr_deltas)])) + 1e-8
t0 = time.perf_counter()
model = train_anomaly_head_phase(model, y_train, delta_sigma)
t1 = time.perf_counter()
print(f"Phase 2: {t1-t0:.1f}s")

print("\n--- Phase 3: GRAC training ---")
tr_preds2, tr_anom_probs = batch_predict_v3(model, X_tr)
adt = AdaptiveDeltaThreshold()
score = 0.0; sc_list=[]; dn_list=[]
for i in range(min(len(y_tr_act), len(tr_preds2))):
    d = abs(float(y_tr_act[i]) - float(tr_preds2[i]))
    if not np.isfinite(d): continue
    thr = adt.threshold()
    is_hard = 1.0 if d > thr else 0.0
    is_soft = float(tr_anom_probs[i]) if i < len(tr_anom_probs) else is_hard
    is_a = ALPHA_BLEND * is_soft + (1-ALPHA_BLEND) * is_hard
    score = BETA * is_a + (1-BETA) * score
    adt.update(d); sc_list.append(score); dn_list.append(d/delta_sigma)
theta_sig = float(np.clip(np.percentile(sc_list, 90), 0.10, 0.80))

sc_arr = np.array(sc_list, dtype=np.float32)
dn_arr = np.array(dn_list, dtype=np.float32)
seq_len = len(sc_arr)
# Construct pseudo-labels
raw = 1.0 / (1.0 + np.exp(-(sc_arr - theta_sig) * GRAC_PSEUDO_SHARP))
labels = np.zeros(seq_len, dtype=np.float32)
ema = 0.0
for t in range(seq_len):
    ema = GRAC_EMA_ALPHA * float(raw[t]) + (1-GRAC_EMA_ALPHA) * ema
    labels[t] = ema

sc_t  = torch.tensor(sc_arr, dtype=torch.float32)
dn_t  = torch.tensor(dn_arr, dtype=torch.float32)
lbl_t = torch.tensor(labels, dtype=torch.float32)

grac = GRAC()
optimizer = torch.optim.Adam(grac.parameters(), lr=GRAC_LR)
criterion = nn.BCEWithLogitsLoss()
grac.train()
t0 = time.perf_counter()
for epoch in range(GRAC_EPOCHS):
    h = torch.zeros(1, GRAC_HIDDEN); optimizer.zero_grad()
    logits = []
    for t in range(seq_len):
        inp = torch.stack([sc_t[t], dn_t[t]]).unsqueeze(0)
        h = grac.gru(inp, h); logits.append(grac.proj(h))
    loss = torch.cat(logits, 0)
    criterion(loss, lbl_t.unsqueeze(1)).backward()
    nn.utils.clip_grad_norm_(grac.parameters(), 1.0); optimizer.step()
    if (epoch+1) % 5 == 0:
        te = time.perf_counter()
        print(f"  Epoch {epoch+1}/{GRAC_EPOCHS} | {te-t0:.1f}s elapsed")

t1 = time.perf_counter()
print(f"Phase 3 (GRAC, {seq_len} steps, {GRAC_EPOCHS} epochs): {t1-t0:.1f}s")
print(f"\nTotal estimated single-sig time: {(t1-t0) + 0:.1f}s + Phase1 + Phase2")
