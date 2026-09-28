---
name: ml-experiment-check
description: Correctness checklist for machine-learning experiment code in this project (California Housing, pandas + scikit-learn). Use when writing, changing or reviewing code that builds X/y, splits data, trains or evaluates a model, computes metrics, compares models or adds features — including notebook cells. Focuses on leakage, baselines, fair comparison, honest evaluation and reproducibility, not on maximising the score.
---

# ML Experiment Check

A checklist for making ML experiments in this project **correct and honest**. Use it before presenting ML code, and when reviewing the learner's code.

The philosophy is **baseline first → simple model → measure → understand the errors → add complexity only when justified**. The goal is understanding and generalization, not leaderboard scores.

## Project conventions (check these first)

| Convention | Value |
|---|---|
| Target | `MedHouseVal`, in units of **$100,000** (report MAE as dollars: 0.40 → ≈ $40,000) |
| Split | `train_test_split(X, y, test_size=0.2, random_state=42)` for every compared experiment |
| First feature subset | `["MedInc", "HouseAge", "AveRooms"]`, later all 8 |
| Baseline | `DummyRegressor(strategy="mean")` evaluated on the same split |
| Primary metric | MAE (+ RMSE and R² as supporting metrics) |
| Data loader | `house_price.data.load_housing_data()` |

## 1. ML correctness

- [ ] `y` is exactly the target column; **the target is not inside `X`** (`"MedHouseVal" not in X.columns`).
- [ ] No feature is derived from the target (price ratios, anything computed from `MedHouseVal`).
- [ ] Every feature would be **available at prediction time** (the prediction-time availability test).
- [ ] `X` and `y` have the same number of rows and are aligned. Row filtering/sorting happens on `df` *before* splitting into `X`/`y`.
- [ ] `X` is 2-D (`df[["col"]]`, not `df["col"]`).
- [ ] **Split before anything that learns from data.** Scalers, imputers, encoders, feature selection and statistics used for features are fitted on the training set only (a `Pipeline` does this automatically; it is *not* required when there is no preprocessing).
- [ ] The test set is never used in `fit()`, for choosing features, or for choosing settings.

## 2. Experiment design

- [ ] A **baseline** exists and is evaluated on the **same** test set as every other model.
- [ ] Models being compared use the **same split** (same `random_state`, same rows) and the same metric code.
- [ ] Each experiment **changes one thing** from a previous one (model *or* features *or* a setting) and states the hypothesis.
- [ ] Results are recorded in an experiment table (model, features, train MAE, test MAE, notes).
- [ ] No automatic hyperparameter searches, mass feature generation, or many-model competitions unless the learner explicitly asks. Tuning with `GridSearchCV` belongs to a later phase.
- [ ] Model choices (`max_depth`, feature sets) are not repeatedly tuned against the test set. Use a validation split or cross-validation once those are introduced.

## 3. Evaluation

- [ ] Metrics are computed on **test** data for model quality. Training metrics are used only to diagnose over/underfitting.
- [ ] **Both** train and test error are reported when judging complexity. A large gap → possible overfitting; both high → underfitting.
- [ ] MAE is translated to dollars. RMSE is compared to MAE: RMSE ≫ MAE means a few large errors, so inspect them.
- [ ] R² is **not** described as "accuracy" or a percentage correct.
- [ ] Argument order is `metric(y_true, y_pred)`.
- [ ] Individual predictions are inspected (actual vs predicted, largest absolute errors), not just summary metrics.
- [ ] Dataset quirks are considered when interpreting errors: `MedHouseVal` is **capped at 5.00001** (965 rows at the exact cap; "5.0" means "$500k or more"), `HouseAge` is capped at 52, and `AveRooms`/`AveBedrms`/`AveOccup`/`Population` have extreme outliers.

## 4. Reproducibility

- [ ] `random_state` is set for the split and for any random model (e.g. `RandomForestRegressor(random_state=42)`).
- [ ] Code runs top-to-bottom in a fresh kernel (no hidden notebook state).
- [ ] Reusable logic lives in `src/house_price/` with a test; exploration stays in notebooks.

## 5. Common wrong claims to correct

- "Lower training error → better model." Only test performance measures generalization.
- "More features always help." Test it (3 vs 8 features on the same split).
- "`fit()` stores the answers." It learns parameters (coefficients, tree splits).
- "Coefficient size = feature importance." It depends on units and on the other features.
- "Random Forest is better, so use it." Only if it beats simpler models on the same test set by enough to justify its complexity.

## How to report problems

For each problem: **what** is wrong (file/cell/line), **why it matters** in plain English (the learner is new to ML), and a **hint** towards the fix. Prefer hints over complete rewritten code for learning milestones.
