# Revision Plan for DAET Journal Paper

## Main Goal

The paper should present itself as:

> a careful empirical comparative evaluation of the DAET detection phase, using Treeherder-alert agreement as a proxy evaluation signal

The paper should **not** read as if it has already proven:

- real debugging effectiveness
- improved engineer productivity
- universally best deployment behavior across workloads

## Main Problems to Fix

These are the revision targets that matter most.

1. The paper still leans too much toward debugging-benefit and deployment-benefit language, even though the actual evidence is proxy-based.
2. The replay experiment currently creates trust risk because it is not fully aligned with the canonical evaluation pipeline.
3. The EWMA result in the canonical experiment is not fully defensible because `alpha = 0.05` is only the best **tested boundary value**, not a confirmed optimum.
4. The paper uses two different ground-truth definitions in different sections, which can confuse reviewers.
5. The ARIMA-order ablation is based on too small a subset to justify strong wording.
6. The paper repeats the same ranking and caveats too often, which weakens the tone.

## Final Priority Order

### Must Do

1. Tighten manuscript framing across abstract, introduction, discussion, and conclusion.
2. Fix the replay inconsistency or clearly demote replay to supplementary deployment evidence.
3. Extend the EWMA tuning grid below `alpha = 0.05` and update the canonical results if needed.
4. Clarify the dual ground-truth setup so readers cannot confuse the main results with the ablation results.
5. Fix writing and repetition issues identified in `writtenreview.md`.

### Good to Do

1. Add a targeted false-positive audit for SMA and EWMA.
2. Add one short explicit explanation of what “false positive” means under Treeherder-agreement evaluation.

### Only if possible (and quick)
1. ARIMA-order expansion, unless it is easy to run cleanly.
2. If ARIMA-order expansion is not done, weaken the wording and caption so the claim remains modest and defensible.


## Section-by-Section Revision Checklist

## 1. Abstract

### Goal

Make the abstract scientifically precise and immediately clear about what the paper actually evaluates.

### Required changes

1. Keep the empirical-extension framing, but tighten the wording so the reader understands this is an evaluation paper, not a new-method paper.
2. Make sure the abstract says early that the metrics reflect agreement with Treeherder alerts, not direct debugging effectiveness or true anomaly validity.
3. Clarify any percentage in the abstract by naming the metric explicitly.

### Specific issue to fix

The sentence:

- `On mozilla-beta, SMA again leads (91.7%), followed closely by EWMA and ARIMA, while LAST remains weakest.`

should name the metric explicitly. For example:

- `On mozilla-beta, SMA again leads in detection rate (91.7%), followed closely by EWMA and ARIMA, while LAST remains weakest.`

### Success condition

A reviewer should be able to read only the abstract and understand:

- the benchmark is Treeherder-alert agreement
- simple statistical methods perform competitively
- practical debugging benefit is not directly proven

## 2. Introduction

### Goal

Frame the problem honestly and establish the paper’s contribution without overclaiming.

### Required changes

1. Add one early sentence stating that the evaluation metrics are based on agreement with Treeherder alerts and therefore act as proxy signals, not direct measures of debugging utility.
2. Add one short paragraph in the introduction explaining that this journal paper extends the earlier conference version and briefly naming the new empirical additions.
3. Soften wording that sounds like the current paper has already validated real deployment effectiveness.


### Success condition

A reviewer should not be able to say:

- “This is just the conference paper with a few extra tables.”

## 3. Related Work

### Goal

Keep this section clean and conventional.

### Required changes

1. Change `Related Works` to `Related Work`.
2. Keep the section concise and avoid overstating how much prior work differs unless the difference is directly relevant to the evaluation story.

### Success condition

The section should read like standard positioning, not like a defense brief.

## 4. Calibration-Baseline Results Section

### Goal

Keep the calibration baseline useful, but make sure it is visibly secondary to the canonical evaluation and internally consistent.

### Required changes

1. Preserve the “calibration baseline only” framing box.
2. Make sure the text does not accidentally present the 70/30 baseline as the main result.
3. Fix the Wilcoxon wording error:
   - change `all ten pairwise comparisons` to `all six pairwise comparisons`
4. Make sure every time this section is used in later discussion, it is explicitly identified as the calibration baseline.

### Success condition

No reviewer should confuse the calibration baseline with the canonical headline experiment.

## 5. Canonical 60/20/20 Evaluation

### Goal

Make this the unquestioned center of the paper.

### Required changes

1. Keep the existing statement that these are the headline results.
2. Recheck all later sections so that canonical results are used whenever the paper makes its strongest scientific claims.
3. Avoid mixing calibration and canonical results in a way that makes cross-protocol comparisons sound cleaner than they are.

### Success condition

If a reviewer asks “what are the main results of this paper?”, the answer should point directly to this section.

## 6. EWMA Tuning Revision

### Goal

Remove the current weakness where `alpha = 0.05` is treated as the selected optimum even though it lies at the lower boundary of the tested grid.

### Required experiment

Extend the EWMA tuning grid below `0.05`.

### Minimum required grid

Use at least:

- `alpha in {0.01, 0.02, 0.03, 0.05, 0.10, 0.15, ..., 0.50}`

If the pipeline already supports arbitrary grids, that is enough. There is no need to redesign the experiment.

### Required procedure

1. Use the same canonical 60/20/20 split already used in the paper.
2. Keep all other settings identical to the current canonical EWMA experiment.
3. Tune `alpha` on the validation set only.
4. Evaluate exactly once on the test set using the selected `alpha`.
5. Do this for both datasets currently in the paper.

### What to update after rerunning

Update all relevant items if the best `alpha` changes:

- validation-tuning figure
- canonical result tables
- discussion text
- conclusion text
- threats to validity text about the EWMA boundary issue

### If the result does not change

If `alpha = 0.05` remains best even after testing smaller values:

- state explicitly that the expanded grid confirmed the result
- remove or revise wording that implies the optimum is still unresolved

### Success condition

After revision, the paper should no longer contain a reviewer-visible gap of the form:

- “Why didn’t you test `alpha < 0.05`?”

## 7. Replay Experiment Revision

### Goal

Remove the trust problem caused by replay using a different evaluation path from the main analysis.

### Preferred option

Re-implement or rerun replay within the same canonical unified pipeline used for the main evaluation.

### Required tasks under the preferred option

1. Identify exactly where replay currently diverges from the canonical evaluation workflow.
2. Move replay input generation and filtering into the same canonical processing path if possible.
3. Recompute:
   - storage reduction
   - trigger frequency
   - alert coverage
4. Recheck all counts:
   - total signatures
   - alerted signatures
   - matched signatures

5. Update all replay tables, figures, and narrative text.

### Fallback option

If full unification cannot be done cleanly without destabilizing the rest of the paper:

1. Keep replay as a supplementary deployment-oriented analysis.
2. Clearly label it as being on a different split or processing path.
3. Reduce its rhetorical weight in the discussion and conclusion.
4. Do not use replay to support the strongest cross-method scientific claims.

### Success condition

There should be no unexplained signature-count discrepancy left in the paper.

## 8. Dual Ground-Truth Clarification

### Goal

Make it impossible for readers to accidentally compare numbers that were computed under different positive-label definitions without realizing it.

### Current issue

The paper uses:

- all Treeherder alerts in the main calibration-baseline table
- regression-classified alerts only in the ablation section

This is explained, but still easy to miss.

### Required revision

Choose one of the following approaches and apply it consistently.

### Option A: Stronger table-level clarification

1. Add a small clarifying row, note, or companion value near the main calibration-baseline results showing the stricter regression-only-GT counterpart for the default setting.
2. Make sure the table caption explicitly says that the ablation uses a stricter GT and therefore values are not directly numerically comparable.

### Option B: Stronger centralized explanation

1. Convert the current boxed note into a short paragraph that appears once, clearly and prominently.
2. Reference that explanation directly in every ablation caption where needed.
3. Make sure the default-setting comparison is stated in one place only and referenced elsewhere.

### Preferred choice

Option A is better if it can be done cleanly, because reviewers often look at tables before reading explanatory prose.

### Success condition

A reader skimming the results section should not mistakenly conclude that the paper is inconsistent just because the same detector has different F1 values in different sections.

## 9. ARIMA Order Ablation Scope

### Goal

Make sure the paper does not overclaim what the ARIMA-order ablation can support.

### Current issue

The ARIMA-order sweep is based on a small subset, which is acceptable as an indicative analysis but weak if presented as decisive evidence.

### Required action

Do one of the following:

### Option A: Expand the subset

If it is easy and safe to rerun, increase the subset size and regenerate the order-comparison results.

### Option B: Downgrade the wording

If the experiment remains limited:

1. change the caption and surrounding text to say the result is indicative
2. avoid language such as:
   - `confirmed`
   - `best order`
   - `demonstrates`
3. use language such as:
   - `suggests`
   - `within this subset`
   - `supports retaining ARIMA(1,1,1) as the reference configuration`

### Success condition

The paper’s wording about ARIMA order should match the strength of the actual experiment.

## 10. False-Positive Audit

### Goal

Add one targeted piece of evidence that goes beyond raw Treeherder-agreement precision.

### Scope

Audit only:

- SMA
- EWMA

These are the right choices because they are the strongest classical methods in the canonical results.

### Required sample design

1. Work on the test set only.
2. Identify cases currently counted as false positives under the paper’s alignment rule.
3. Randomly sample a manageable subset from SMA and EWMA.
4. Keep the sampling procedure simple and documented.

### For each sampled case, inspect:

1. the local time-series pattern around the detection
2. whether the detection corresponds to a visible excursion or regime change
3. whether there is any supporting Bugzilla regression signal if such linking is feasible

### Labels to assign

For each sampled false positive, assign one of:

- likely genuine anomaly
- likely noise
- unclear

### Required outputs

1. A short methods paragraph describing the sampling and labeling procedure.
2. A compact summary table with counts and percentages for the three labels.
3. A small number of representative examples, only if they are genuinely informative.
4. One discussion paragraph explaining what the audit implies for interpreting low precision.

### Important constraint

This is not meant to become a full new benchmark or large annotation project. Keep it focused and directly tied to the paper’s precision interpretation.

### Success condition

The paper gains one concrete piece of supporting evidence for the claim that some “false positives” may be genuine but unlabelled anomalies.

## 11. Precision Explanation

### Goal

Make the low precision easier for reviewers to interpret without sounding defensive.

### Required changes

1. Add one short paragraph near the main results explaining that low precision here means low agreement with Treeherder alerts, not necessarily low true-anomaly precision.
2. Keep this explanation concise.
3. Avoid repeating the same caveat too many times across the paper.

### Success condition

The paper acknowledges the limitation clearly without sounding like it is apologizing on every page.

## 12. Discussion and Conclusion Compression

### Goal

Reduce repetition and improve confidence of tone.

### Required changes

1. Identify repeated restatements of:
   - the method ranking
   - the Treeherder-agreement caveat
   - the journal-extension contribution summary
2. Keep each major point in full form only where it matters most.
3. In later sections, refer back to earlier sections instead of fully restating the same claim.

### Specific compression targets

Compress repetition in:

- discussion
- threats to validity
- conclusion

### Success condition

The paper should feel deliberate and controlled, not repetitive or defensive.

## If Time Gets Tight

Apply this fallback priority exactly.

### Must keep

1. Framing rewrite.
2. Replay fix or replay demotion.
3. EWMA retuning.
4. Ground-truth clarification.

### Good to keep

1. False-positive audit for SMA and EWMA.

### Can be lighter

1. ARIMA-order expansion, unless it is easy.
2. If ARIMA-order expansion is not done, soften the claim rather than spending more scope on it.





## Submission Venues

The current intended submission venues are:

1. `Information and Software Technology (IST)`
2. `Performance Evaluation`
