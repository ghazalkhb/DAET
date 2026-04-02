"""
Compute V3 Transformer Bootstrap CI and Wilcoxon tests vs classical methods.
Also extract exact runtime/memory for V3 Transformer.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon as wilcoxon_test
import json, os, sys

BASE = r"d:\Naser\2\results"

# ── Load evaluation_all.csv ──────────────────────────────────────────────────
eval_path = os.path.join(BASE, "evaluation_all.csv")
df = pd.read_csv(eval_path)
print("Columns:", df.columns.tolist())
print("Methods:", df["method"].unique().tolist() if "method" in df.columns else "N/A")
print("Shape:", df.shape)
print(df.head(3))

# Normalise column name
method_col = "method" if "method" in df.columns else "Method"

# ── Separate by method ───────────────────────────────────────────────────────
# Detect metric columns
prec_col = [c for c in df.columns if "precision" in c.lower()][0]
rec_col  = [c for c in df.columns if "recall" in c.lower()][0]
f1_col   = [c for c in df.columns if c.lower() == "f1"][0]
far_col  = [c for c in df.columns if "false_alarm" in c.lower() or "far" in c.lower()][0]

print(f"\nUsing cols: prec={prec_col}, rec={rec_col}, f1={f1_col}, far={far_col}")

methods = df[method_col].unique().tolist()
print("Methods found:", methods)

# Find Transformer method name
tf_name = [m for m in methods if "transform" in m.lower()]
print("Transformer name:", tf_name)
tf_name = tf_name[0] if tf_name else None

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return lo, hi

print("\n=== BOOTSTRAP CI (V3 Transformer) ===")
if tf_name:
    tf_df = df[df[method_col] == tf_name]
    for col, name in [(prec_col, "Precision"), (rec_col, "Recall"),
                      (f1_col, "F1"), (far_col, "FAR")]:
        vals = tf_df[col].dropna().values
        lo, hi = bootstrap_ci(vals)
        print(f"  {name}: [{lo:.3f}, {hi:.3f}]  (n={len(vals)})")

# Also print all methods CI for comparison
print("\n=== BOOTSTRAP CI ALL METHODS ===")
for m in methods:
    mdf = df[df[method_col] == m]
    p_lo, p_hi = bootstrap_ci(mdf[prec_col].dropna().values)
    r_lo, r_hi = bootstrap_ci(mdf[rec_col].dropna().values)
    f_lo, f_hi = bootstrap_ci(mdf[f1_col].dropna().values)
    fa_lo, fa_hi = bootstrap_ci(mdf[far_col].dropna().values)
    print(f"  {m}: P=[{p_lo:.3f},{p_hi:.3f}]  R=[{r_lo:.3f},{r_hi:.3f}]  F1=[{f_lo:.3f},{f_hi:.3f}]  FAR=[{fa_lo:.4f},{fa_hi:.4f}]")

# ── Wilcoxon tests: V3 Transformer vs each classical method ──────────────────
classical = [m for m in methods if "transform" not in m.lower()]
print("\n=== WILCOXON: V3 Transformer vs Classical Methods ===")
if tf_name:
    tf_df = df[df[method_col] == tf_name].sort_values("signature_id" if "signature_id" in df.columns else df.columns[0])
    for cls in classical:
        cls_df = df[df[method_col] == cls].sort_values("signature_id" if "signature_id" in df.columns else df.columns[0])
        # Align on common signatures
        common = set(tf_df.iloc[:,0]) & set(cls_df.iloc[:,0])
        tf_a = tf_df[tf_df.iloc[:,0].isin(common)].sort_values(tf_df.columns[0])
        cl_a = cls_df[cls_df.iloc[:,0].isin(common)].sort_values(cls_df.columns[0])
        print(f"\n  {tf_name} vs {cls}  (n={len(common)} common sigs):")
        for col, name in [(prec_col, "Precision"), (rec_col, "Recall"),
                          (f1_col, "F1"), (far_col, "FAR")]:
            a = tf_a[col].values
            b = cl_a[col].values
            try:
                stat, pval = wilcoxon_test(a, b, zero_method='wilcox')
                sig = "" if pval < 0.05 else " (ns)"
                print(f"    {name}: p={pval:.4g}{sig}  (TF_mean={a.mean():.4f} cls_mean={b.mean():.4f})")
            except Exception as e:
                print(f"    {name}: error {e}")

# ── Also compute Detection Rate and mean delay for V3 ────────────────────────
print("\n=== V3 Transformer Summary Statistics ===")
if tf_name:
    tf_df = df[df[method_col] == tf_name]
    print(f"  n_signatures = {len(tf_df)}")
    for col in [prec_col, rec_col, f1_col, far_col]:
        vals = tf_df[col].dropna().values
        print(f"  {col}: mean={vals.mean():.4f}  std={vals.std():.4f}")
    
    # Check for detection rate / delay columns
    extra_cols = [c for c in tf_df.columns if any(x in c.lower() for x in
                  ["detect", "delay", "storage", "runtime", "memory", "peak"])]
    for col in extra_cols:
        vals = tf_df[col].dropna().values
        try:
            print(f"  {col}: mean={vals.mean():.4f}  std={vals.std():.4f}")
        except:
            print(f"  {col}: {tf_df[col].unique()[:5]}")

# ── Try to get runtime from summary.json ─────────────────────────────────────
summary_path = os.path.join(BASE, "summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summ = json.load(f)
    print("\n=== SUMMARY.JSON ===")
    for k, v in summ.items():
        if "transform" in k.lower():
            print(f"  {k}: {json.dumps(v, indent=4)}")

# ── Check compute metrics file ────────────────────────────────────────────────
# Look for any file with runtime info
for fn in os.listdir(BASE):
    if "compute" in fn.lower() or "runtime" in fn.lower() or "timing" in fn.lower():
        print(f"\nFound file: {fn}")
        fpath = os.path.join(BASE, fn)
        if fn.endswith(".csv"):
            tmp = pd.read_csv(fpath)
            print(tmp.to_string())
        elif fn.endswith(".json"):
            with open(fpath) as f:
                print(json.dumps(json.load(f), indent=2)[:2000])

print("\nDone.")
