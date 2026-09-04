# Transformer statistical comparison results

This analysis compares the Transformer with the two strongest
classical baselines, SMA and EWMA. The canonical shared-gate analysis and the
detector-specific-gate robustness analysis are reported separately.

## Statistical design

- Unit of analysis: signature (342 `firefox-android`; 1,477 `mozilla-beta`).
- Transformer handling: average its five seeded runs within each signature
  before inferential testing. The seeds are repeated measurements, not five
  independent signatures.
- Planned contrasts: Transformer minus SMA and Transformer minus EWMA.
- Outcomes: precision, recall, F1, and false-alert rate (FAR).
- Test: two-sided paired Wilcoxon signed-rank test.
- Multiplicity: Holm correction within each protocol/dataset family over the
  eight planned tests (two comparators by four outcomes).
- Interval estimate: 95% paired bootstrap confidence interval for the mean
  difference, based on 10,000 signature resamples with base seed 42.
- Effect size: matched-pairs rank-biserial correlation. Positive values favor
  Transformer for precision, recall, and F1; negative values favor Transformer
  for FAR.

## Primary F1 results

Differences are Transformer minus classical. Positive F1 differences favor the
Transformer.

| Protocol | Dataset | Comparator | Transformer | Classical | Difference | 95% paired-bootstrap CI | Holm-adjusted p | Rank-biserial | Conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Shared gate | firefox-android | SMA | 0.1051 | 0.1039 | +0.0012 | [-0.0102, 0.0127] | 1.0000 | -0.030 | no detectable difference |
| Shared gate | firefox-android | EWMA | 0.1051 | 0.1033 | +0.0018 | [-0.0078, 0.0115] | 1.0000 | -0.011 | no detectable difference |
| Shared gate | mozilla-beta | SMA | 0.0789 | 0.0847 | -0.0058 | [-0.0110, -0.0006] | 0.000018 | -0.277 | Transformer lower |
| Shared gate | mozilla-beta | EWMA | 0.0789 | 0.0830 | -0.0041 | [-0.0092, 0.0012] | 0.000159 | -0.244 | distributional difference; mean CI overlaps zero |
| Detector-specific gate | firefox-android | SMA | 0.1462 | 0.2138 | -0.0676 | [-0.0945, -0.0412] | 0.000003 | -0.580 | Transformer lower |
| Detector-specific gate | firefox-android | EWMA | 0.1462 | 0.2031 | -0.0569 | [-0.0825, -0.0321] | 0.000013 | -0.538 | Transformer lower |
| Detector-specific gate | mozilla-beta | SMA | 0.1007 | 0.1102 | -0.0094 | [-0.0181, -0.0006] | 0.024908 | -0.150 | Transformer lower |
| Detector-specific gate | mozilla-beta | EWMA | 0.1007 | 0.1089 | -0.0082 | [-0.0171, 0.0008] | 0.024908 | -0.143 | distributional difference; mean CI overlaps zero |

The shared-gate result does not support an F1 advantage for the Transformer.
Its small numerical lead on `firefox-android` is statistically indistinguishable
from SMA and EWMA. On `mozilla-beta`, the Transformer is lower than both under
the Wilcoxon analysis. Detector-specific gate tuning strengthens the classical
methods and places the Transformer below both SMA and EWMA on both datasets.

## Other outcomes

Under the shared gate, Transformer recall is higher than EWMA on
`firefox-android` (Holm p = 0.0111), but its FAR is also higher than both SMA and
EWMA (both Holm p < 2.1e-11). On `mozilla-beta`, Transformer precision and FAR
are worse than both baselines, while recall does not differ detectably.

Under detector-specific gates, Transformer precision and FAR are worse than
both baselines on both datasets. Its recall is not detectably different on
`firefox-android` and is higher on `mozilla-beta` (both Holm p < 3.8e-24), but
this recall increase does not offset its precision/FAR disadvantage in F1.

## Interpretation cautions

Approximately 69--75% of signature-level F1 differences are exactly zero, so
the median paired F1 difference is zero in every contrast. The Wilcoxon test
evaluates the signed-rank distribution among nonzero pairs, whereas the
bootstrap interval estimates the mean difference. Consequently, a significant
Wilcoxon result can coexist with a mean-difference interval that overlaps zero,
as occurs for the two EWMA comparisons on `mozilla-beta`. Those cases should be
described as distributional evidence, not as a bootstrap-confirmed nonzero mean
difference.

The descriptive seed analysis reaches the same practical pattern: all five
Transformer seeds are below SMA and EWMA on `mozilla-beta`, and all five are
below both comparators under detector-specific tuning on both datasets. Under
the canonical shared gate on `firefox-android`, the tiny difference changes
sign across seeds, reinforcing the conclusion that there is no stable F1
advantage there.

## Files

- `paired_comparisons.csv`: all 32 inferential comparisons and diagnostics.
- `method_bootstrap_ci.csv`: 48 method/outcome mean estimates and intervals.
- `transformer_seed_sensitivity.csv`: descriptive per-seed F1 sensitivity.
- `summary.json`: compact machine-readable F1 results and analysis design.
- `run_metadata.json`: input hashes and software environment.
- `f1_paired_difference_forest.png`: F1 differences with paired-bootstrap CIs.

Reproduce with:

```bash
python Experiments/transformer_statistical_comparison.py
python Experiments/validate_transformer_statistics.py
```
