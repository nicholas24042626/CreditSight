# Aligning the KNIME Workflow with the Python XGBoost Pipeline

Comparison based on `python/train_models.py` and `python/common.py` (the tuned XGBoost
pipeline: winsorization + Optuna-tuned hyperparameters, ~0.6995 accuracy / 0.6067 macro F1).

## Missing entirely

### 1. Winsorization (biggest gap)
The KNIME flow has no clipping node. In Python this is the single largest individual
accuracy gain found in the ablation work (`common.py`, `Winsorizer` class):

- Clip each numeric column to its 1st/99th percentile.
- Bounds must be learned on the **training partition only**, then applied to the test
  partition (Learner/Predictor pattern — not one global node over the whole dataset).
- KNIME equivalent: **Numeric Outliers** node, percentile-based replace strategy, as a
  Learner (fit on train) → Apply (transform test) pair, inserted right after the Table
  Partitioner.

## Wrong / extra steps

### 2. Normalizer / Normalizer (Apply) — remove
Python's XGBoost config uses `scale_numeric=False` — no MinMax/z-score scaling. Tree
ensembles don't need it, and it makes the KNIME model diverge from the tuned Python one.
Remove both Normalizer nodes and replace that spot with the Winsorizer Learner/Apply pair
from item 1.

### 3. Missing Value node runs before the Table Partitioner — reorder
Python fits imputation (median for numeric, most-frequent for categorical) only on the
training split via `SimpleImputer` inside a `ColumnTransformer`, then applies it to test —
never fit on combined data. Doing it before partitioning in KNIME leaks test-set statistics
into training.

Fix: move imputation to *after* the partitioner, as a Missing Value (Learner on train) →
Missing Value (Apply on test) pair, mirroring the Winsorizer.

### 4. Second Missing Value node after XGBoost Predictor, before Scorer
No equivalent in the Python pipeline. Suggests predictions or a column is coming out with
missing values — usually a symptom of a mismatch upstream (e.g. One to Many producing
different category columns between train/test branches). Should resolve once items 1-3 are
fixed; if not, investigate as a real bug.

### 5. Duplicate Row Filter removes duplicates — Python doesn't
`common.py` only *counts* duplicates for reporting (`duplicate_rows: df.duplicated().sum()`),
it never drops them. Remove this node for an exact match, unless divergence here is
intentional.

## Needs verification (probably fine, confirm details)

### 6. Rule Engine
Should implement the exact rating → group mapping (`common.py`, `RATING_GROUP_MAPPING`):

| Rating(s) | Group |
|---|---|
| AAA, AA, A | Investment-High |
| BBB | Investment-Low |
| BB, B | Speculative |
| CCC, CC, C, D | Distressed |

Anything else (unrated, blank) should be dropped, matching `dropna(subset=["RatingGroup"])`.

### 7. Column Filter
Should drop identifier columns only: name, company/issuer/security name, symbol, ticker,
date, rating agency name, plus the raw rating column itself. Nothing else should be dropped.

### 8. Table Partitioner
Must be **stratified**, 70% train / 30% test, matching:
```python
train_test_split(..., test_size=0.30, random_state=42, stratify=y_encoded)
```
Set partitioning method to stratified sampling on `RatingGroup` with a fixed random seed.
KNIME's RNG won't reproduce Python's split row-for-row, but the ratio/stratification logic
should match.

## XGBoost hyperparameters — set exactly, don't rely on the Parameter Optimization Loop

These are the final Optuna-tuned values, not something to re-search in KNIME:

| Parameter | Value |
|---|---|
| n_estimators | 324 |
| max_depth | 8 |
| learning_rate | 0.08110932021587948 |
| subsample | 0.8212663057499048 |
| colsample_bytree | 0.6684815593910434 |
| min_child_weight | 1 |
| gamma | 0.009096143390093125 |
| reg_lambda | 0.5098610905755356 |
| num_parallel_tree | 3 |
| tree_method | hist |
| max_bin | 16 |
| eval_metric | mlogloss |
| random_state | 42 |

If you keep the Parameter Optimization Loop to independently re-tune in KNIME, that's a
legitimate separate search — but don't expect it to reproduce the Python numbers (accuracy
0.6995 / macro F1 0.6067). For parity with the Python pipeline, hard-code these values in
the XGBoost Tree Ensemble Learner node and drop the loop.

## Priority order

1. Add Winsorization (Learner/Apply pair after partitioning).
2. Remove the Normalizer nodes.
3. Move Missing Value imputation to after partitioning (Learner/Apply pair).
4. Lock in the exact XGBoost hyperparameters (skip/bypass the optimization loop).
5. Verify Rule Engine, Column Filter, and Table Partitioner settings (items 6-8).
6. Re-check the Duplicate Row Filter and the second Missing Value node once the above is
   fixed.
