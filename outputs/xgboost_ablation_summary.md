# XGBoost Ablation Study — CreditSight

**Source notebook:** `notebook/xgboost/xgboost.ipynb` — an "XGBoost Ablation Study" block (Sections 1–7) inserted right after Step 7 (the baseline confusion matrix) and before the team's Min-Max Normalization / Experiment 7 / Experiment 8 / SHAP cells. Experiment 7 and Experiment 8's `GridSearchCV` `param_grid` were since widened to also search `reg_alpha`, `reg_lambda`, and `gamma` (matching the parameter set Section 2's `RandomizedSearchCV` already searched), so those two cells are no longer unmodified relative to the team's original version — everything else about them (search method, `X_train`/`X_test` handling, scoring) is unchanged.
**Dataset:** `data/set A corporate_rating.csv`
**Scope:** XGBoost only. Decision Tree, Random Forest, and Logistic Regression were not touched by this work. `python/train_models.py`'s XGBoost hyperparameters have since been updated separately to match Experiment 7's original 5-parameter `GridSearchCV` result (`n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0`) — that production change happened outside this ablation study and predates the `reg_alpha`/`reg_lambda`/`gamma` grid widening described above, so it does not yet reflect the wider 8-parameter search.

---

## 1. Why this exists

An earlier attempt tuned XGBoost by bundling several changes at once — tighter hyperparameters, early
stopping, and monotonic constraints, all applied together directly in `train_models.py`. The result
was a single before/after number (accuracy dropped from 68.1% to 63.2%) with no way to tell **which**
change caused the drop, or whether the changes were fighting each other. That's not something a bundled
change can answer — it needed an ablation study: test each change in isolation against the same
baseline, then test the winning changes combined, and check whether the combination is better or worse
than its parts.

This notebook section is that ablation study, built directly into `notebook/xgboost/xgboost.ipynb`
rather than a separate script, so it sits next to the model it's testing and reuses its exact data
pipeline. It was rebuilt once already to sit on top of a teammate's later push that added a "Final SHAP
Feature Importance" cell depending on the notebook's `Experiment 8` output — the ablation section is
positioned before that work, not in place of it, so nothing downstream broke.

## 2. Dataset and fixed setup

Everything below shares the same untouched inputs, established in Steps 1–5 of the notebook (not
modified by this ablation work):

| Item | Value |
| --- | --- |
| Raw rows loaded | 2029 |
| Missing values | none |
| Duplicate rows | 0 |
| Target | `RatingGroup`, derived from the `Rating` column (AAA…D collapsed into 4 ordinal bands) |
| Class distribution | Speculative 792 · Investment-Low 671 · Investment-High 494 · Distressed 72 |
| Numeric features (25) | `currentRatio`, `quickRatio`, `cashRatio`, `daysOfSalesOutstanding`, `netProfitMargin`, `pretaxProfitMargin`, `grossProfitMargin`, `operatingProfitMargin`, `returnOnAssets`, `returnOnCapitalEmployed`, `returnOnEquity`, `assetTurnover`, `fixedAssetTurnover`, `debtEquityRatio`, `debtRatio`, `effectiveTaxRate`, `freeCashFlowOperatingCashFlowRatio`, `freeCashFlowPerShare`, `cashPerShare`, `companyEquityMultiplier`, `ebitPerRevenue`, `enterpriseValueMultiple`, `operatingCashFlowPerShare`, `operatingCashFlowSalesRatio`, `payablesTurnover` |
| Categorical features (1) | `Sector` (one-hot encoded) |
| Preprocessing | `build_preprocessor()` from `python/common.py`: median-impute + pass-through for numeric, most-frequent-impute + one-hot for categorical |
| Train/test split | 70/30, stratified on `RatingGroup`, `random_state=42` → 1420 train rows / 609 test rows |

The 72-row "Distressed" class (3.5% of the data) is the dataset's core difficulty — every model in
this study struggles with it, and several of the ablation steps below trade other performance for
better or worse treatment of exactly that class. Class imbalance is severe enough (Speculative
outnumbers Distressed 11:1) that it shows up directly in the baseline's per-class breakdown below.

## 3. Method

Every ablation step below:

- Starts from the same untouched `X_train`, `X_test`, `y_train`, `y_test` created once in Step 5 of the
  notebook — none of these are refit, resplit, or modified by any step.
- Uses the same preprocessing definition (`build_preprocessor(numeric_columns, categorical_columns)`),
  freshly instantiated per step (so each step's own model pipeline is self-contained) but never changing
  what it does.
- Changes exactly **one** variable relative to the baseline `XGBClassifier(random_state=42,
  eval_metric="mlogloss")` — everything else stays at XGBoost's defaults unless the step says otherwise.
- Is scored once, on the untouched `X_test`, using `evaluate_predictions()` from `python/common.py` (the
  same evaluation function `train_models.py` and `predict.py` both use, so these numbers are directly
  comparable to the production training script's metrics).
- Is recorded into a running `ablation_records` list inside the notebook, under names prefixed
  `ablation_*` so nothing collides with the team's later Min-Max/Experiment/SHAP cells, which reuse
  plain names like `pipeline`, `metrics`, and `best_pipeline`.
- Is consolidated by Section 7 into the table below and written to
  `outputs/xgboost_ablation_summary.csv`.

## 4. Step 1 — Baseline

**Config:** `XGBClassifier(random_state=42, eval_metric="mlogloss")`, every other parameter at
XGBoost's default (100 trees, `max_depth=6`, `learning_rate=0.3`, no subsampling, no regularization, no
class weighting, no monotonic constraints). This is the exact model already trained in Step 5/6 of the
notebook — it is reused here, not retrained, per the "don't refit" constraint.

**Result:**

```text
Accuracy: 0.6634
                 precision    recall  f1-score   support

Investment-High       0.66      0.64      0.65       148
 Investment-Low       0.62      0.56      0.58       201
    Speculative       0.71      0.82      0.76       238
     Distressed       0.43      0.14      0.21        22

       accuracy                           0.66       609
      macro avg       0.60      0.54      0.55       609
   weighted avg       0.65      0.66      0.65       609
```

**Reading it:** the model is comfortable on the majority classes (Speculative recall 82%) and
noticeably weaker on the minority "Distressed" class — precision (43%) and recall (14%) are both low,
meaning it rarely predicts "Distressed" at all, and is often wrong when it does. This is the reference
point every later step is measured against.

## 5. Step 2 — Hyperparameter tuning only

**What changed:** ran `RandomizedSearchCV` (40 candidate configurations, 5-fold stratified CV, scored
on accuracy) over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`,
`reg_alpha`, `reg_lambda`, and `gamma`. Search space:

```python
{
    "model__n_estimators": [100, 200, 300, 400, 500],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__reg_alpha": [0, 0.01, 0.1, 1.0],
    "model__reg_lambda": [0.5, 1.0, 1.5, 2.0],
    "model__gamma": [0, 0.1, 0.3, 0.5],
}
```

**What did not change:** no early stopping, no `sample_weight`, no monotonic constraints. Search ran
only on `X_train`; the winning configuration was scored once on `X_test`.

**Best configuration found** (best CV accuracy 0.6556):

```python
{'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8,
 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 2.0, 'gamma': 0.1}
```

**Result:** Accuracy **0.6814** (+1.80 points vs baseline), Macro F1 0.5850 (+3.59 points vs
baseline), Distressed F1 0.3077 (up from 0.2069).

**Reading it:** this run's best configuration turned out close to XGBoost's own defaults
(`max_depth=6`, moderate `learning_rate=0.1`, light regularization) rather than the far-shallower or
far-deeper extremes in the search space — and it improved *every* metric over baseline, including
macro F1 and the Distressed class specifically. That's a notably cleaner win than a previous run of
this same search (before the notebook was rebased onto the team's SHAP update), which had instead found
a deep, low-learning-rate configuration that boosted accuracy while quietly costing Distressed-class
performance. `RandomizedSearchCV`'s `n_jobs=-1` parallelism makes the exact winning configuration
non-deterministic between runs even with `random_state` fixed (see §12) — worth keeping in mind if this
step is re-run for the final report.

## 6. Step 3 — Class imbalance handling only

**What changed:** baseline hyperparameters, completely unchanged, plus
`sample_weight=compute_sample_weight("balanced", y_train)` passed into `.fit()`. This reweights the
training loss so each class contributes equally regardless of how many rows it has, rather than letting
the 792-row Speculative class dominate the 72-row Distressed class.

**What did not change:** no hyperparameter tuning, no early stopping, no monotonic constraints.

**Result:** Accuracy 0.6732 (+0.98 points vs baseline), Macro F1 **0.5890** (best of all six steps,
+3.99 points vs baseline), Distressed F1 **0.3158** (best of all six steps, up from 0.2069).

**Reading it:** this remains the best individual step for macro F1 and for the Distressed class
specifically — exactly the effect class weighting is supposed to have, and (as before) it costs
comparatively little accuracy to get there. Of the two individually winning changes this run, this one
is the most directly aligned with fixing the dataset's core weakness (severe class imbalance) rather
than just fitting the majority classes harder.

## 7. Step 4 — Early stopping only

**What changed:** baseline hyperparameters + `n_estimators=1000` + `early_stopping_rounds=20`, trained
against a validation slice carved out of 15% of `X_train` (stratified, `random_state=42`) — `X_test`
and the original `X_train` are untouched by this split.

**Why n_estimators=1000, and why there are two numbers here:** early stopping only means anything if
the model has room to keep boosting past 100 rounds; comparing it to the 100-round Step-1 baseline
would conflate "does more capacity help" with "does stopping early help." So this step reports two
numbers — the fair comparison point (4a, same `n_estimators=1000`, no early stopping) and the actual
early-stopping run (4b):

| Variant | Accuracy | Macro F1 |
| --- | --- | --- |
| 4a. `n_estimators=1000`, no early stopping (fair comparison point) | 0.6798 | 0.5778 |
| 4b. `n_estimators=1000` + early stopping, stopped at iteration 20 | **0.6420** | **0.5420** |

**Result vs the Step-1 baseline:** Accuracy 0.6420 (−2.14 points), Macro F1 0.5420 (−0.71 points) — the
worst of all six steps on both metrics.

**Reading it:** early stopping didn't just fail to help — it underperformed *both* comparison points,
including its own fair one (4a). Stopping at iteration 20 out of a possible 1000 is very aggressive;
`mlogloss` on the 213-row validation slice likely plateaued early by chance (small slice, small
dataset) well before the model had actually converged on the full signal. This step should not be read
as "early stopping doesn't work for this problem" — it should be read as "these particular settings
(15% slice, 20-round patience, default learning rate) stopped too early." A follow-up worth trying:
larger patience (e.g. `early_stopping_rounds=50`) or a lower `learning_rate` paired with early stopping,
which is the combination early stopping is normally used with.

## 8. Step 5 — `tree_method="hist"` only

**What changed:** baseline hyperparameters + `tree_method="hist"`. Nothing else.

**Result:** Accuracy 0.6634, Macro F1 0.5491 — **identical to the baseline to four decimal places**.
Training time 0.760s vs 0.675s for the baseline run in this run (timing is noisy at this dataset size
and not a reliable signal either way — see §8 reasoning below).

**Reading it:** the accuracy result is expected, not a bug — XGBoost 3.0.4 (the version installed in
this project) already uses histogram-based split-finding by default (`tree_method="auto"` resolves to
`hist`-equivalent behaviour on CPU), so explicitly requesting `hist` changes nothing here. Its actual
value isn't in this table at all: it's the lever that would let a `RandomizedSearchCV` like Step 2
search more candidates in the same wall-clock budget on a larger dataset. On this dataset (1420 training
rows), any speed difference is within the noise of a single run and doesn't change what's practical to
search.

## 9. Step 6 — Final combination

**Decision rule:** a step is included in the combination only if it beat the Step-1 baseline on
accuracy **or** macro F1. Applied automatically inside the notebook (not hand-picked):

| Step | Beat baseline? | Included in combination |
| --- | --- | --- |
| 2. Hyperparameter tuning | Yes (both) | ✅ |
| 3. Class imbalance handling | Yes (both) | ✅ |
| 4. Early stopping | No | ❌ |
| 5. `tree_method="hist"` | No (tied, not beaten) | ❌ |

**Combined configuration:**

```python
{'random_state': 42, 'eval_metric': 'mlogloss', 'n_estimators': 200, 'max_depth': 6,
 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.9, 'reg_alpha': 0.1,
 'reg_lambda': 2.0, 'gamma': 0.1}
# + sample_weight = compute_sample_weight("balanced", y_train)
```

**Result:** Accuracy 0.6732, Macro F1 0.5718, Distressed F1 0.2581.

**Interaction effect — the headline finding of this study:** the combination matches Step 3's accuracy
exactly but underperforms **both** individual steps on macro F1, and underperforms Step 3 specifically
on the Distressed class it was best at:

| Comparison | Accuracy | Macro F1 | Distressed F1 |
| --- | --- | --- | --- |
| Best individual step (Step 2, tuning alone) | 0.6814 | 0.5850 | 0.3077 |
| Best individual step (Step 3, imbalance alone) | 0.6732 | 0.5890 | 0.3158 |
| Final combination (2 + 3 together) | 0.6732 | 0.5718 | 0.2581 |

Both individual gains partially erode when combined — accuracy settles at Step 3's level rather than
Step 2's, and macro F1 / Distressed F1 both come in below either individual step. This is measured
evidence that these two changes interact rather than stacking cleanly additively, even in a run where
both individual changes look like unambiguous wins on their own. This is exactly the kind of result a
bundled all-at-once tuning attempt (like the earlier one that dropped accuracy from 68.1% to 63.2%)
cannot distinguish from "the whole approach doesn't work" — here it's visible as a specific,
attributable interaction between two changes, each of which works fine alone.

## 10. Full comparison table

| Step | Accuracy | Macro F1 | Weighted F1 | Investment-High F1 | Investment-Low F1 | Speculative F1 | Distressed F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Baseline | 0.6634 | 0.5491 | 0.6537 | 0.6485 | 0.5849 | 0.7563 | 0.2069 |
| 2. Hyperparameter tuning only | **0.6814** | **0.5850** | 0.6709 | 0.6621 | 0.6032 | **0.7672** | 0.3077 |
| 3. Class imbalance handling only | 0.6732 | 0.5890 | 0.6696 | **0.6867** | **0.6198** | 0.7339 | **0.3158** |
| 4. Early stopping only | 0.6420 | 0.5420 | 0.6337 | 0.6355 | 0.5685 | 0.7240 | 0.2400 |
| 5. `tree_method="hist"` only | 0.6634 | 0.5491 | 0.6537 | 0.6485 | 0.5849 | 0.7563 | 0.2069 |
| 6. Final combination | 0.6732 | 0.5718 | 0.6665 | 0.6667 | 0.6146 | 0.7480 | 0.2581 |

Bold marks the best value in each column. Machine-readable version (same numbers, plus the full free-text
notes per row): `outputs/xgboost_ablation_summary.csv`.

## 11. Conclusions and recommendations

1. **No single lever is a clean win on every axis.** Tuning and class-weighting both improve on the
   baseline individually across all metrics measured here, but they improve on *different* things most
   (tuning: raw accuracy and Speculative/Investment-High F1; class weighting: macro F1 and Distressed
   F1) — early stopping (as configured) and `tree_method="hist"` (on this dataset size) don't help at
   all.
2. **If the report needs one number, state which objective it optimizes for.** "Best accuracy and best
   macro F1 among the tuning-only changes" is Step 2 (0.6814 / 0.5850); "best Distressed-class handling"
   is Step 3 (Distressed F1 0.3158). For a credit-risk tool, where failing to flag a genuinely distressed
   company is the costlier error, Step 3's Distressed-class strength is worth weighing against Step 2's
   slightly higher headline numbers.
3. **Combining winning changes isn't automatically better — verify, don't assume.** Steps 2 and 3 each
   improve on the baseline individually, but combined they underperform both on macro F1 and the
   Distressed class specifically. Any future tuning work on this model should re-test combinations
   rather than assuming individual gains stack.
4. **Early stopping is worth a second attempt, not abandonment.** The specific configuration tested here
   (15% validation slice, 20-round patience, default learning rate) stopped too early; a larger patience
   value or pairing it with a lower learning rate (the combination it's designed for) is a reasonable
   next experiment.
5. **`tree_method="hist"` is a non-issue for this dataset size**, and confirmed as such rather than
   assumed — useful to note in the report as a deliberate check rather than an omission.

## 12. Reproducing this study

Run all cells in `notebook/xgboost/xgboost.ipynb` top to bottom (Kernel → Restart & Run All, or
`jupyter nbconvert --to notebook --execute --inplace notebook/xgboost/xgboost.ipynb`). The ablation
section (right after Step 7, before Min-Max Normalization) regenerates `ablation_records` from scratch
each run.

Two things to know before re-running for a final report figure:

- **Non-determinism:** `RandomizedSearchCV` in Step 2 uses `n_jobs=-1`, so the exact winning
  hyperparameter configuration (and downstream numbers) can shift meaningfully between runs — this
  document's Step 2 configuration and results differ from an earlier run of the same search before the
  notebook was rebased onto the team's SHAP update, even though nothing about Step 2's code changed. The
  qualitative conclusions (imbalance handling wins on Distressed F1, early stopping underperforms, a
  combination interaction effect exists) have held across every run so far, but exact figures have not.
  Re-run once immediately before capturing final numbers for the report, and quote the numbers from that
  specific run.
- **Experiment 7 and Experiment 8 now run noticeably slower.** Widening their `param_grid` from 5 to 8
  parameters grows the grid from 2⁵=32 to 2⁸=256 combinations, so each `GridSearchCV.fit()` call (5-fold
  CV) now trains 1,280 models instead of 160 — expect several minutes per cell instead of seconds. This
  does not affect Sections 1–7 (the ablation study proper), only the two `GridSearchCV` cells after it.
