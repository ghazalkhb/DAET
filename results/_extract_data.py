import pandas as pd
import json

out = []

df = pd.read_csv('results/evaluation_all.csv')
out.append('=== evaluation_all.csv ===')
avail = [c for c in ['Precision','Recall','F1','FAR','False_Alarm_Rate','Storage_Reduction','Enabled_Fraction'] if c in df.columns]
out.append(df.groupby('method')[avail].mean().round(4).to_string())

with open('results/summary.json') as f:
    s = json.load(f)
out.append('\n=== summary.json ===')
for m, v in s.get('aggregate_metrics', {}).items():
    out.append(f'  [{m}]')
    for k, val in v.items():
        out.append(f'    {k}: {val}')

with open('results/stats_significance.json') as f:
    ss = json.load(f)
bci = ss.get('bootstrap_ci', {})
out.append('\n=== Bootstrap CI ===')
for m, v in bci.items():
    out.append(f'  {m}: Precision={v.get("Precision",{})}, Recall={v.get("Recall",{})}, F1={v.get("F1",{})}, FAR={v.get("FAR",{})}')
wil = ss.get('wilcoxon', {})
out.append('\n=== Wilcoxon p-values ===')
for pair, v in wil.items():
    out.append(f'  {pair}: {v}')

out.append('\n=== Ablation tables ===')
for fn in ['ablation_beta','ablation_threshold','ablation_tau','ablation_sma_window','ablation_arima_order']:
    df2 = pd.read_csv(f'results/{fn}.csv')
    out.append(f'\n--- {fn} ---')
    out.append(df2.to_string(index=False))

df_r = pd.read_csv('results/replay/replay_results.csv')
out.append('\n=== Replay aggregate ===')
out.append(df_r.groupby('method')[[c for c in df_r.columns if c not in ['signature_id','method']]].mean().round(4).to_string())

with open('results/three_way_summary.json') as f:
    tw = json.load(f)
out.append('\n=== Three-way summary ===')
out.append(json.dumps(tw, indent=2))

with open('results/_extracted_data.txt', 'w') as f:
    f.write('\n'.join(out))
print('Done')
