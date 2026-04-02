"""Copy Transformer and comparison plots to journal images directory."""
import os
import shutil

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC   = os.path.join(ROOT, "results", "plots")
DST_R = os.path.join(ROOT, "Journal Paper", "journal_v2", "images", "results_plots")
DST_I = os.path.join(ROOT, "Journal Paper", "journal_v2", "images")

os.makedirs(DST_R, exist_ok=True)
os.makedirs(DST_I, exist_ok=True)

for f in os.listdir(SRC):
    src_path = os.path.join(SRC, f)
    if "transformer" in f.lower():
        shutil.copy2(src_path, os.path.join(DST_R, f))
        print(f"Copied to results_plots: {f}")
    if "comparison" in f or "detection_rate" in f:
        shutil.copy2(src_path, os.path.join(DST_I, f))
        print(f"Copied to images: {f}")

print("Done.")
