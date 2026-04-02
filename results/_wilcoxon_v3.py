"""Compute V3 Transformer Wilcoxon tests and write to JSON output."""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import json

df = pd.read_csv(r'd:\Naser\2\results\evaluation_all.csv')
tf = df[df.method=='Transformer'].sort_values('signature_id').reset_index(drop=True)
cols = ['Precision','Recall','F1','False_Alarm_Rate']
results = {}

for cls in ['ARIMA','LAST','SMA','EWMA']:
    cdf = df[df.method==cls].sort_values('signature_id').reset_index(drop=True)
    match = (tf.signature_id.values == cdf.signature_id.values).all()
    entry = {'n': len(tf), 'id_match': bool(match)}
    for c in cols:
        a, b = tf[c].values, cdf[c].values
        try:
            _, p = wilcoxon(a, b)
            entry[c] = float(p)
        except Exception as e:
            entry[c] = str(e)
    results[f'Transformer_vs_{cls}'] = entry

# Detection rate
n_alerted = int((tf.Num_Alerts_Test > 0).sum())
n_det = int((tf[tf.Num_Alerts_Test>0]['Was_Alert_Detected']=='Yes').sum())
det_rate = n_det / n_alerted * 100

results['summary'] = {
    'n_alerted_sigs': n_alerted,
    'n_detected': n_det,
    'detection_rate_pct': det_rate,
    'total_runtime_s': float(tf.Runtime_ms.sum() / 1000),
    'mean_peak_mem_kb': float(tf.Peak_Mem_KB.mean()),
}

out_path = r'd:\Naser\2\results\_v3_wilcoxon.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Written to {out_path}")
for k, v in results.items():
    print(k, json.dumps(v, indent=2))
