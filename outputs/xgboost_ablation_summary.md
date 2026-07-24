# XGBoost Ablation Study — CreditSight

**Source notebook:** `notebook/xgboost/xgboost.ipynb` — an "XGBoost Ablation Study" block (Sections 1–8) inserted right after Step 7 (the baseline confusion matrix) and before the team's Min-Max Normalization / Experiment 7 / Experiment 8 / SHAP cells. Section 6 (scoring objective) was added later, replacing two ad hoc "Improvement 2" diagnostic cells that used to sit after Experiment 7/8 instead of inside the ablation study proper. Experiment 7 and Experiment 8 (later in the same notebook) use their own, separately-evolved `GridSearchCV` grid — not covered by this report.
**Dataset:** `data/set A corporate_rating.csv`
**Scope:** XGBoost only. Decision Tree, Random Forest, and Logistic Regression were not touched by this work. `python/train_models.py`'s XGBoost hyperparameters have since been updated separately to match Experiment 7's `GridSearchCV` result — that production change happened outside this ablation study.
**Numbers below** are from a full, fresh top-to-bottom execution of the notebook (`jupyter nbconvert --to notebook --execute --inplace`), not hand-edited — see §13 for the non-determinism caveat this implies.

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
- Is consolidated by Section 8 into the table below and written to
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
on accuracy) over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.
Search space:

```python
{
    "model__n_estimators": [100, 200, 300, 400, 500],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}
```

**What did not change:** no early stopping, no `sample_weight`, no monotonic constraints, no
regularization parameters (`reg_alpha`/`reg_lambda`/`gamma` stay at XGBoost's defaults — only the five
parameters above are searched). Search ran only on `X_train`; the winning configuration was scored once
on `X_test`.

**Best configuration found** (best CV accuracy 0.6648):

```python
{'subsample': 0.7, 'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 1.0}
```

**Result:** Accuracy **0.6798** (+1.64 points vs baseline), Macro F1 0.5800 (+3.09 points vs
baseline), Distressed F1 0.2857 (up from 0.2069).

**Reading it:** the search found a deeper tree (`max_depth=8` vs the default 6) with a lower learning
rate than the default (0.1 vs 0.3) and moderate subsampling — a fairly standard "more capacity, more
regularization via sampling" combination, and it improved every metric over baseline. `RandomizedSearchCV`'s
`n_jobs=-1` parallelism makes the exact winning configuration non-deterministic between runs even with
`random_state` fixed (see §13) — the specific hyperparameters above are this run's result, not a fixed
constant.

## 6. Step 3 — Class imbalance handling only

**What changed:** baseline hyperparameters, completely unchanged, plus
`sample_weight=compute_sample_weight("balanced", y_train)` passed into `.fit()`. This reweights the
training loss so each class contributes equally regardless of how many rows it has, rather than letting
the 792-row Speculative class dominate the 72-row Distressed class.

**What did not change:** no hyperparameter tuning, no early stopping, no monotonic constraints.

**Result:** Accuracy 0.6732 (+0.98 points vs baseline), Macro F1 **0.5890** (best of all seven
individual/scoring steps, +3.99 points vs baseline), Distressed F1 **0.3158**, up from 0.2069.

**Reading it:** this remains the best individual step for macro F1 and for the Distressed class
specifically — exactly the effect class weighting is supposed to have, and it costs comparatively little
accuracy to get there. This step is deterministic (no random search involved), so its numbers are stable
across re-runs.

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
worst of all seven individual/scoring steps on both metrics.

**Reading it:** early stopping didn't just fail to help — it underperformed *both* comparison points,
including its own fair one (4a). Stopping at iteration 20 out of a possible 1000 is very aggressive;
`mlogloss` on the 213-row validation slice likely plateaued early by chance (small slice, small
dataset) well before the model had actually converged on the full signal. This step should not be read
as "early stopping doesn't work for this problem" — it should be read as "these particular settings
(15% slice, 20-round patience, default learning rate) stopped too early." A follow-up worth trying:
larger patience (e.g. `early_stopping_rounds=50`) or a lower `learning_rate` paired with early stopping,
which is the combination early stopping is normally used with. This step is deterministic and matched
exactly across the previous and current runs of this study.

## 8. Step 5 — `tree_method="hist"` only

**What changed:** baseline hyperparameters + `tree_method="hist"`. Nothing else.

**Result:** Accuracy 0.6634, Macro F1 0.5491 — **identical to the baseline to four decimal places**.
Training time 0.295s vs 0.301s for the baseline run in this run (timing is noisy at this dataset size
and not a reliable signal either way).

**Reading it:** the accuracy result is expected, not a bug — XGBoost 3.0.4 (the version installed in
this project) already uses histogram-based split-finding by default (`tree_method="auto"` resolves to
`hist`-equivalent behaviour on CPU), so explicitly requesting `hist` changes nothing here. Its actual
value isn't in this table at all: it's the lever that would let a `RandomizedSearchCV` like Step 2
search more candidates in the same wall-clock budget on a larger dataset. On this dataset (1420 training
rows), any speed difference is within the noise of a single run and doesn't change what's practical to
search. This step is deterministic and matched exactly across runs.

## 9. Step 6 — Scoring objective: `scoring="f1_macro"` only

**What changed:** identical setup to Step 2 (same `RandomizedSearchCV` grid, `n_iter=40`, 5-fold
`StratifiedKFold`) — the only variable changed is the search's `scoring` argument, from `"accuracy"` to
`"f1_macro"`. This isolates the effect of the search objective itself, independent of every other
choice. It replaces two ad hoc "Improvement 2" diagnostic cells that previously tested this same idea
against Experiment 7/8's own separately-tuned baselines instead of the shared Step-1 baseline used by
every other section here.

**Best configuration found** (best CV macro F1 0.5698):

```python
{'subsample': 0.7, 'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 1.0}
```

**Result:** Accuracy 0.6798, Macro F1 0.5800, Distressed F1 0.2857 — **identical to Step 2's result to
four decimal places.**

**Reading it — this is the headline finding of this section:** scoring on `f1_macro` instead of
`accuracy` selected the *exact same* hyperparameter configuration as Step 2. On this grid and this
dataset, the search objective alone does not change which hyperparameters `RandomizedSearchCV` lands on
— a clean, verified negative result, not an assumption. This does not mean scoring objective never
matters (Section 3's `sample_weight`-based imbalance handling still shows the value of directly
targeting the minority class), only that *for this specific 5-parameter grid*, accuracy and macro F1
apparently agree closely enough across candidates that both scoring functions rank the same winner top.

## 10. Step 7 — Final combination

**Decision rule:** a step is included in the combination only if it beat the Step-1 baseline on
accuracy **or** macro F1. Applied automatically inside the notebook (not hand-picked). Step 6 (scoring
objective) is not a candidate for inclusion here — it's a diagnostic on Step 2's search, not an
independent lever, so including it would double-count Step 2's hyperparameters.

| Step | Beat baseline? | Included in combination |
| --- | --- | --- |
| 2. Hyperparameter tuning | Yes (both) | ✅ |
| 3. Class imbalance handling | Yes (both) | ✅ |
| 4. Early stopping | No | ❌ |
| 5. `tree_method="hist"` | No (tied, not beaten) | ❌ |

**Combined configuration:**

```python
{'random_state': 42, 'eval_metric': 'mlogloss', 'subsample': 0.7, 'n_estimators': 200,
 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 1.0}
# + sample_weight = compute_sample_weight("balanced", y_train)
```

**Result:** Accuracy **0.6831**, Macro F1 **0.6063**, Distressed F1 **0.3750** — the best result in
the table on every metric, including the Distressed class.

**No interaction effect this run — a different result from an earlier run of this study.** An earlier
execution of this same ablation study found that combining Steps 2 and 3 *underperformed* both
individual steps on macro F1 and Distressed F1 (an interaction effect, documented as this report's
previous headline finding). In this fresh run, the combination instead **improves on both individual
steps on every metric measured**:

| Comparison | Accuracy | Macro F1 | Distressed F1 |
| --- | --- | --- | --- |
| Step 2 (tuning alone) | 0.6798 | 0.5800 | 0.2857 |
| Step 3 (imbalance alone) | 0.6732 | 0.5890 | 0.3158 |
| Step 7 (2 + 3 combined) | **0.6831** | **0.6063** | **0.3750** |

This reversal is itself the most important methodological point in this report: **whether these two
changes interact or stack additively is not a fixed property of the model — it varies between runs**,
because Step 2's search is non-deterministic (`RandomizedSearchCV` with `n_jobs=-1`, see §13). A single
run's combination result is not reliable evidence either way; only a repeated-run or repeated-split
methodology (as later experiments in the notebook, past this ablation study, adopt via a 15-split
harness) can distinguish "these changes interact" from "this run's search landed somewhere that happened
not to combine well."

## 11. Full comparison table

| Step | Accuracy | Macro F1 | Weighted F1 | Investment-High F1 | Investment-Low F1 | Speculative F1 | Distressed F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Baseline | 0.6634 | 0.5491 | 0.6537 | 0.6485 | 0.5849 | 0.7563 | 0.2069 |
| 2. Hyperparameter tuning only | 0.6798 | 0.5800 | 0.6711 | 0.6526 | 0.6221 | 0.7597 | 0.2857 |
| 3. Class imbalance handling only | 0.6732 | 0.5890 | 0.6696 | 0.6867 | 0.6198 | 0.7339 | 0.3158 |
| 4. Early stopping only | 0.6420 | 0.5420 | 0.6337 | 0.6355 | 0.5685 | 0.7240 | 0.2400 |
| 5. `tree_method="hist"` only | 0.6634 | 0.5491 | 0.6537 | 0.6485 | 0.5849 | 0.7563 | 0.2069 |
| 6. Scoring objective (f1_macro) only | 0.6798 | 0.5800 | 0.6711 | 0.6526 | 0.6221 | 0.7597 | 0.2857 |
| 7. Final combination | **0.6831** | **0.6063** | **0.6773** | **0.6779** | 0.6146 | 0.7579 | **0.3750** |

Bold marks the best value in each column. Machine-readable version (same numbers, plus the full free-text
notes per row): `outputs/xgboost_ablation_summary.csv`.

## 12. Conclusions and recommendations

1. **The scoring objective alone doesn't change the search outcome, on this grid.** Step 6 found the
   identical hyperparameters as Step 2 despite optimizing a different metric (`f1_macro` vs `accuracy`)
   — a genuine ablation result, not an assumption. Don't extrapolate this to "scoring never matters":
   Section 3's `sample_weight` approach shows targeting the minority class directly still works; it's
   specifically the *search objective* axis, isolated from every other change, that made no difference
   here.
2. **Whether combined changes interact is not a fixed property — it can flip between runs.** This run's
   Step 7 combination *improved* on both individual steps on every metric; an earlier run of the same
   study found the opposite (an interaction effect where the combination underperformed both). Both
   results are real, correctly-measured outcomes of the same code — the difference is `RandomizedSearchCV`
   non-determinism in Step 2 propagating downstream. **Report which run a headline number comes from, and
   don't treat a single run's combination result as conclusive** — a repeated-split methodology (used
   later in the same notebook, past this ablation study) is what settles this properly.
3. **If the report needs one number, state which run and which objective it optimizes for.** In this
   run, Step 7 (the combination) is the best result on every metric, including the Distressed class
   (F1 0.3750) — a stronger conclusion than either individual step alone gave in this run, though see
   point 2 above before treating that as guaranteed to reproduce.
4. **Early stopping is worth a second attempt, not abandonment.** The specific configuration tested here
   (15% validation slice, 20-round patience, default learning rate) stopped too early; a larger patience
   value or pairing it with a lower learning rate (the combination it's designed for) is a reasonable
   next experiment. This step's result has been identical across every run of this study so far (it's
   deterministic).
5. **`tree_method="hist"` is a non-issue for this dataset size**, and confirmed as such rather than
   assumed — identical to baseline across every run of this study so far.

## 13. Reproducing this study

Run all cells in `notebook/xgboost/xgboost.ipynb` top to bottom (Kernel → Restart & Run All, or
`jupyter nbconvert --to notebook --execute --inplace notebook/xgboost/xgboost.ipynb`). The ablation
section (right after Step 7, before Min-Max Normalization) regenerates `ablation_records` from scratch
each run.

**Non-determinism, demonstrated, not just warned about:** `RandomizedSearchCV` in Step 2 (and Step 6,
which shares its grid) uses `n_jobs=-1`, so the exact winning hyperparameter configuration can shift
between runs even with `random_state` fixed. This is not hypothetical — it is the documented reason
this report's §10 "no interaction effect" finding differs from an earlier run's "interaction effect"
finding, using the exact same code both times. Steps 1, 3, 4, and 5 are deterministic and have matched
exactly across every run so far; only Steps 2, 6, and 7 (which depend on Step 2's search) vary. Re-run
once immediately before capturing final numbers for any report, and quote the numbers from that specific
run — including which finding (interaction effect present or absent) that run produced.
