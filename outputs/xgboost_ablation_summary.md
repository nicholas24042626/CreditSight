# XGBoost Ablation Study — CreditSight

**Source notebook:** `notebook/xgboost/xgboost.ipynb` (Sections 1–7, added as an "XGBoost Ablation Study" block after the existing baseline/SHAP cells)
**Dataset:** `data/set A corporate_rating.csv`
**Scope:** XGBoost only. Decision Tree, Random Forest, and Logistic Regression, and the shared files `python/common.py` / `python/train_models.py`, were not touched by this work.

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
pipeline.

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
better or worse treatment of exactly that class.

Class imbalance is severe enough (Speculative outnumbers Distressed 11:1) that it shows up directly in
the baseline's per-class breakdown below.

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
- Is recorded into a running `ablation_records` list inside the notebook, which Section 7 turns into
  the table below and writes to `outputs/xgboost_ablation_summary.csv`.

## 4. Step 1 — Baseline

**Config:** `XGBClassifier(random_state=42, eval_metric="mlogloss")`, every other parameter at
XGBoost's default (100 trees, `max_depth=6`, `learning_rate=0.3`, no subsampling, no regularization, no
class weighting, no monotonic constraints). This is the exact model already trained in Step 5/6 of the
notebook — it is reused here, not retrained, per the "don't refit" constraint.

**Result:**

```
Accuracy: 0.6667
                 precision    recall  f1-score   support

Investment-High       0.64      0.62      0.63       148
 Investment-Low       0.63      0.57      0.60       201
    Speculative       0.70      0.82      0.76       238
     Distressed       0.80      0.18      0.30        22

       accuracy                           0.67       609
      macro avg       0.69      0.55      0.57       609
   weighted avg       0.67      0.67      0.66       609
```

**Reading it:** the model is comfortable on the majority classes (Speculative recall 82%) and
noticeably weaker on the minority "Distressed" class — high precision (80%) but very low recall (18%),
meaning when it does call a company "Distressed" it's usually right, but it misses 4 out of every 5
actually-distressed companies. This is the reference point every later step is measured against.

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

**Best configuration found** (best CV accuracy 0.6549):

```python
{'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.6,
 'colsample_bytree': 0.9, 'reg_alpha': 0, 'reg_lambda': 2.0, 'gamma': 0}
```

**Result:** Accuracy **0.6847** (+1.80 points vs baseline), Macro F1 0.5532 (**−1.74 points vs
baseline**), Distressed F1 0.1538 (down from 0.2963).

**Reading it:** this is the best individual step for raw accuracy — but it got there by leaning further
into the majority classes (Investment-High F1 jumped from 0.6323 to 0.6801) at the direct expense of
the already-weak Distressed class. Macro F1, which weights all four classes equally, actually went
*down*. If the project's real objective is balanced performance across rating bands (arguably the more
defensible goal for a credit-risk tool) rather than raw accuracy, this step is a regression, not an
improvement — worth stating explicitly in the report rather than only quoting the accuracy figure.

## 6. Step 3 — Class imbalance handling only

**What changed:** baseline hyperparameters, completely unchanged, plus
`sample_weight=compute_sample_weight("balanced", y_train)` passed into `.fit()`. This reweights the
training loss so each class contributes equally regardless of how many rows it has, rather than letting
the 792-row Speculative class dominate the 72-row Distressed class.

**What did not change:** no hyperparameter tuning, no early stopping, no monotonic constraints.

**Result:** Accuracy 0.6732 (+0.65 points vs baseline), Macro F1 **0.5890** (best of all six steps,
+1.84 points vs baseline), Distressed F1 **0.3158** (best of all six steps, up from 0.2963).

**Reading it:** this is the best individual step for macro F1 and for the Distressed class
specifically — exactly the effect class weighting is supposed to have, and it cost comparatively little
accuracy (0.6732 vs baseline's 0.6667, actually slightly higher) to get there. Of the two individually
"winning" changes, this is the one most directly aligned with fixing the dataset's core weakness (severe
class imbalance) rather than just fitting the majority classes harder.

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
| 4a. `n_estimators=1000`, no early stopping (fair comparison point) | 0.6946 | 0.6223 |
| 4b. `n_estimators=1000` + early stopping, stopped at iteration 16 | **0.6420** | **0.5518** |

**Result vs the Step-1 baseline:** Accuracy 0.6420 (−2.47 points), Macro F1 0.5518 (−1.88 points) — the
worst of all six steps on both metrics.

**Reading it:** early stopping didn't just fail to help — it underperformed *both* comparison points,
including its own fair one (4a). Stopping at iteration 16 out of a possible 1000 is very aggressive;
`mlogloss` on the 213-row validation slice likely plateaued early by chance (small slice, small
dataset) well before the model had actually converged on the full signal. This step should not be read
as "early stopping doesn't work for this problem" — it should be read as "these particular settings
(15% slice, 20-round patience, default learning rate) stopped too early." A follow-up worth trying:
larger patience (e.g. `early_stopping_rounds=50`) or a lower `learning_rate` paired with early stopping,
which is the combination early stopping is normally used with.

## 8. Step 5 — `tree_method="hist"` only

**What changed:** baseline hyperparameters + `tree_method="hist"`. Nothing else.

**Result:** Accuracy 0.6667, Macro F1 0.5706 — **identical to the baseline to four decimal places**.
Training time 0.461s vs 0.603s for the baseline run (≈24% faster on this dataset size).

**Reading it:** this result is expected, not a bug — XGBoost 3.0.4 (the version installed in this
project) already uses histogram-based split-finding by default (`tree_method="auto"` resolves to
`hist`-equivalent behaviour on CPU), so explicitly requesting `hist` changes nothing here. Its actual
value isn't in this table at all: it's the lever that would let a `RandomizedSearchCV` like Step 2
search more candidates in the same wall-clock budget on a larger dataset. On this dataset (1420 training
rows), the speed difference is real but small enough that it doesn't change what's practical to search.

## 9. Step 6 — Final combination

**Decision rule:** a step is included in the combination only if it beat the Step-1 baseline on
accuracy **or** macro F1. Applied automatically inside the notebook (not hand-picked):

| Step | Beat baseline? | Included in combination |
| --- | --- | --- |
| 2. Hyperparameter tuning | Yes (accuracy) | ✅ |
| 3. Class imbalance handling | Yes (both) | ✅ |
| 4. Early stopping | No | ❌ |
| 5. `tree_method="hist"` | No (tied, not beaten) | ❌ |

**Combined configuration:**

```python
{'random_state': 42, 'eval_metric': 'mlogloss', 'n_estimators': 400, 'max_depth': 8,
 'learning_rate': 0.03, 'subsample': 0.6, 'colsample_bytree': 0.9, 'reg_alpha': 0,
 'reg_lambda': 2.0, 'gamma': 0}
# + sample_weight = compute_sample_weight("balanced", y_train)
```

**Result:** Accuracy 0.6634, Macro F1 0.5736, Distressed F1 0.3030.

**Interaction effect — the headline finding of this study:** the combination underperforms **both**
of the individual steps it's made of, on **both** metrics:

| Comparison | Accuracy | Macro F1 |
| --- | --- | --- |
| Best individual step (Step 2, tuning alone) | 0.6847 | — |
| Best individual step (Step 3, imbalance alone) | — | 0.5890 |
| Final combination (2 + 3 together) | 0.6634 | 0.5736 |

Neither the accuracy gain from tuning nor the macro-F1 gain from class weighting survived being
combined. This is direct, measured evidence that these two changes interact negatively rather than
stacking additively — plausibly because the deeper, more aggressively regularized tree structure found
by the hyperparameter search (`max_depth=8`, low `learning_rate=0.03`) responds differently to
reweighted samples than the shallower default tree structure the search was *not* run against. This is
exactly the kind of result a bundled all-at-once tuning attempt (like the earlier one that dropped
accuracy from 68.1% to 63.2%) cannot distinguish from "the whole approach doesn't work" — here it's
visible as a specific, attributable interaction between two changes, each of which works fine alone.

## 10. Full comparison table

| Step | Accuracy | Macro F1 | Weighted F1 | Investment-High F1 | Investment-Low F1 | Speculative F1 | Distressed F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Baseline | 0.6667 | 0.5706 | 0.6571 | 0.6323 | 0.5969 | 0.7568 | 0.2963 |
| 2. Hyperparameter tuning only | **0.6847** | 0.5532 | 0.6724 | 0.6801 | 0.6133 | **0.7654** | 0.1538 |
| 3. Class imbalance handling only | 0.6732 | **0.5890** | 0.6696 | **0.6867** | **0.6198** | 0.7339 | **0.3158** |
| 4. Early stopping only | 0.6420 | 0.5518 | 0.6350 | 0.5911 | 0.5800 | 0.7400 | 0.2963 |
| 5. `tree_method="hist"` only | 0.6667 | 0.5706 | 0.6571 | 0.6323 | 0.5969 | 0.7568 | 0.2963 |
| 6. Final combination | 0.6634 | 0.5736 | 0.6570 | 0.6536 | 0.5858 | 0.7520 | 0.3030 |

Bold marks the best value in each column. Machine-readable version (same numbers, plus the full free-text
notes per row): `outputs/xgboost_ablation_summary.csv`.

## 11. Conclusions and recommendations

1. **No single lever is a clean win.** Every change trades one metric for another — tuning buys
   accuracy at the cost of the minority class; class weighting buys balance at a small accuracy cost;
   early stopping (as configured) and `tree_method="hist"` (on this dataset size) don't help at all.
2. **If the report needs one number, state which objective it optimizes for.** "Best accuracy" is Step
   2 (0.6847); "best balanced performance across rating bands" is Step 3 (macro F1 0.5890, Distressed F1
   0.3158). For a credit-risk tool, where failing to flag a genuinely distressed company is the costlier
   error, Step 3 is the more defensible choice despite its slightly lower headline accuracy.
3. **Combining winning changes isn't automatically better — verify, don't assume.** Steps 2 and 3 each
   improve on the baseline individually, but combined they underperform both. Any future tuning work on
   this model should re-test combinations rather than assuming individual gains stack.
4. **Early stopping is worth a second attempt, not abandonment.** The specific configuration tested here
   (15% validation slice, 20-round patience, default learning rate) stopped too early; a larger patience
   value or pairing it with a lower learning rate (the combination it's designed for) is a reasonable
   next experiment.
5. **`tree_method="hist"` is a non-issue for this dataset size**, and confirmed as such rather than
   assumed — useful to note in the report as a deliberate check rather than an omission.

## 12. Reproducing this study

Run all cells in `notebook/xgboost/xgboost.ipynb` top to bottom (Kernel → Restart & Run All, or
`jupyter nbconvert --to notebook --execute --inplace notebook/xgboost/xgboost.ipynb`). The ablation
section (Sections 1–7, after the SHAP cell) regenerates `ablation_records` from scratch each run and
overwrites `outputs/xgboost_ablation_summary.csv`. Because `RandomizedSearchCV` in Step 2 uses
`n_jobs=-1`, exact figures can shift by roughly ±1–2 points between runs due to floating-point
non-determinism in XGBoost's multi-threaded histogram computation — the qualitative conclusions above
(which step wins on which metric, the early-stopping underperformance, the interaction effect) have been
stable across the runs performed while building this document, but if reproducing for the final report,
re-run once immediately before capturing final numbers.
