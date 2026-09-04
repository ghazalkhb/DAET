import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('combined_detail.csv')
np.random.seed(42)
B = 2000

def boot_ci(arr, B=2000):
    n = len(arr)
    means = [np.mean(np.random.choice(arr, n, replace=True)) for _ in range(B)]
    return np.mean(arr), np.percentile(means, 2.5), np.percentile(means, 97.5)

# ---- Firefox-Android ----
fa = df[df['dataset']=='firefox-android'].copy()
# Average Transformer seeds per signature
tr_fa = fa[fa['method'].str.startswith('Transformer')].groupby('signature_id')[
    ['test_precision','test_recall','test_f1','test_far']].mean().reset_index()
tr_fa['method'] = 'Transformer'
fa_stat = fa[~fa['method'].str.startswith('Transformer')]

print("=== FIREFOX-ANDROID BOOTSTRAP CIs (n=342 signatures each) ===")
methods_fa = ['ARIMA(1,1,1)', 'LAST', 'SMA(w=40)', 'EWMA(a=0.05)']
fa_scores = {}
for m in methods_fa:
    sub = fa_stat[fa_stat['method']==m]
    fa_scores[m] = {}
    for metric in ['test_precision','test_recall','test_f1','test_far']:
        arr = sub[metric].values
        mn, lo, hi = boot_ci(arr)
        fa_scores[m][metric] = (mn, lo, hi)
        print(f"  {m} {metric}: mean={mn:.4f} CI=[{lo:.4f},{hi:.4f}]")

print()
print("=== FIREFOX-ANDROID WILCOXON (n=342) ===")
pairs = [
    ('ARIMA(1,1,1)','LAST'),
    ('ARIMA(1,1,1)','SMA(w=40)'),
    ('ARIMA(1,1,1)','EWMA(a=0.05)'),
    ('LAST','SMA(w=40)'),
    ('LAST','EWMA(a=0.05)'),
    ('SMA(w=40)','EWMA(a=0.05)'),
]
fa_wilcoxon = {}
for m1, m2 in pairs:
    key = f"{m1} vs {m2}"
    fa_wilcoxon[key] = {}
    s1 = fa_stat[fa_stat['method']==m1]
    s2 = fa_stat[fa_stat['method']==m2]
    for metric in ['test_precision','test_recall','test_f1','test_far']:
        a = s1.set_index('signature_id')[metric]
        b = s2.set_index('signature_id')[metric]
        common = a.index.intersection(b.index)
        try:
            res = stats.wilcoxon(a.loc[common].values, b.loc[common].values)
            sig = '* sig' if res.pvalue < 0.05 else '  ns'
            fa_wilcoxon[key][metric] = res.pvalue
            print(f"  {key} {metric}: p={res.pvalue:.6f}{sig}")
        except Exception as e:
            print(f"  {key} {metric}: ERROR {e}")

# ---- Mozilla-beta ----
mb = df[df['dataset']=='mozilla-beta'].copy()
mb_stat = mb[~mb['method'].str.startswith('Transformer')]

print()
print("=== MOZILLA-BETA BOOTSTRAP CIs (n=1477 signatures each) ===")
methods_mb = ['ARIMA(1,1,1)', 'LAST', 'SMA(w=20)', 'EWMA(a=0.05)']
mb_scores = {}
for m in methods_mb:
    sub = mb_stat[mb_stat['method']==m]
    mb_scores[m] = {}
    for metric in ['test_precision','test_recall','test_f1','test_far']:
        arr = sub[metric].values
        mn, lo, hi = boot_ci(arr)
        mb_scores[m][metric] = (mn, lo, hi)
        print(f"  {m} {metric}: mean={mn:.4f} CI=[{lo:.4f},{hi:.4f}]")

print()
print("=== MOZILLA-BETA WILCOXON (n=1477) ===")
pairs_mb = [
    ('ARIMA(1,1,1)','LAST'),
    ('ARIMA(1,1,1)','SMA(w=20)'),
    ('ARIMA(1,1,1)','EWMA(a=0.05)'),
    ('LAST','SMA(w=20)'),
    ('LAST','EWMA(a=0.05)'),
    ('SMA(w=20)','EWMA(a=0.05)'),
]
mb_wilcoxon = {}
for m1, m2 in pairs_mb:
    key = f"{m1} vs {m2}"
    mb_wilcoxon[key] = {}
    s1 = mb_stat[mb_stat['method']==m1]
    s2 = mb_stat[mb_stat['method']==m2]
    for metric in ['test_precision','test_recall','test_f1','test_far']:
        a = s1.set_index('signature_id')[metric]
        b = s2.set_index('signature_id')[metric]
        common = a.index.intersection(b.index)
        try:
            res = stats.wilcoxon(a.loc[common].values, b.loc[common].values)
            sig = '* sig' if res.pvalue < 0.05 else '  ns'
            mb_wilcoxon[key][metric] = res.pvalue
            print(f"  {key} {metric}: p={res.pvalue:.6f}{sig}")
        except Exception as e:
            print(f"  {key} {metric}: ERROR {e}")

print()
print("DONE")
