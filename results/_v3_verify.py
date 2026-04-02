# -*- coding: utf-8 -*-
"""
V3 quick verification: run on first 10 signatures only (fastest to slowest),
dump results to a temp CSV to verify V3 logic without waiting for all 214.
"""
import os, sys, time, json
os.chdir(r'd:\Naser\2')
sys.path.insert(0, r'd:\Naser\2\Code')

import numpy as np, pandas as pd, torch, torch.nn as nn, math, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

# ── Constants (copy from V3's __file__ without import) ──────────────────────
WINDOW=30; D_MODEL=32; N_HEAD=4; N_LAYERS=2; DIM_FEEDFORWARD=128
DROPOUT=0.1; LR=5e-4; WEIGHT_DECAY=1e-4; HUBER_DELTA=1.0
EPOCHS=30; BATCH_SIZE=128; PATIENCE=8; MAX_TRAIN_WIN=600
BETA=0.30; THRESHOLD_ANOMALY=0.30; K_SIGMA=3.0; MIN_THRESHOLD=1e-6
TOL=5; TRAIN_RATIO=0.70; PLOT_N=0
ALPHA_MU=0.15; ALPHA_VAR=0.10; THETA_PERCENTILE=90
ANOM_HEAD_HIDDEN=16; ANOM_EPOCHS=8; ANOM_PATIENCE=3; ANOM_LR=5e-4
ALPHA_BLEND=0.60; MIN_ANOM_TRAIN=30
GRAC_HIDDEN=16; GRAC_EPOCHS=10; GRAC_PATIENCE=3; GRAC_LR=5e-4
GRAC_THRESHOLD=0.50; GRAC_PSEUDO_SHARP=8.0; GRAC_EMA_ALPHA=0.40
MIN_GRAC_TRAIN=20

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device("cpu")

N_SIGS = 10  # test on first 10 signatures only

VECTORS_DIR = r'd:\Naser\2\results\vectors'
ALERTS_CSV  = r'd:\Naser\2\Data\alerts_data.csv'

# ── Load V3 components but DON'T trigger the main block ─────────────────────
# We redefine the necessary classes here to avoid importing the module
exec(open(r'd:\Naser\2\Code\run_transformer_v3.py', encoding='utf-8').read().split(
    "\n# ═══════════════════════════════════════════════════════════════════════════════\n# Main\n"
)[0])

print("V3 classes loaded. Testing on first", N_SIGS, "signatures...")

# Load vectors
avail = []
for f in os.listdir(VECTORS_DIR):
    if f.endswith(".npy"):
        y = np.load(os.path.join(VECTORS_DIR, f)).astype(float)
        avail.append((f[:-4], y))
avail.sort(key=lambda x: -len(x[1]))
sigs_y = avail[:N_SIGS]
print(f"Loaded {len(sigs_y)} vectors for testing")

# Run V3
t0_global = time.perf_counter()
df_v3, costs = run_transformer_split(sigs_y)
elapsed = time.perf_counter() - t0_global

print(f"\nDone. {len(sigs_y)} sigs in {elapsed:.1f}s ({elapsed/len(sigs_y):.1f}s/sig)")
print(f"Rows: {len(df_v3)}, sigs: {df_v3['signature_id'].nunique()}")
print(f"isAnomaly dtype: {df_v3['isAnomaly'].dtype}")
print(f"isAnomaly has floats: {(df_v3['isAnomaly'] - df_v3['isAnomaly'].astype(int)).abs().max():.4f}")
print(f"Sample isAnomaly values: {df_v3['isAnomaly'].head(10).tolist()}")
print(f"decision=1 fraction: {(df_v3['decision']==1).mean():.3f}")
print(f"Mean anomalyScore: {df_v3['anomalyScore'].mean():.4f}")

df_v3.to_csv(r'd:\Naser\2\results\_v3_sample.csv', index=False)
print(f"\nSaved to results/_v3_sample.csv")
