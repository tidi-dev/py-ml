# 06 — Evaluating a Model: How Wrong Is It, Really?

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → **06 Evaluation** → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** The project code doesn't do this yet. Read it now for orientation, or when you reach this milestone. Do the exercises when you get here in the project.

| Where we are | |
|---|---|
| **Earlier** | We trained a baseline ([chapter 04](04-first-baseline-model.md)) and a Linear Regression ([chapter 05](05-linear-regression.md)), and got predictions for the test set. |
| **Now** | Turn "predictions vs. real answers" into a few numbers that tell us how good a model is, and compare models fairly. |
| **Next** | Discover why good results on *training* data can be a trap ([chapter 07](07-overfitting-and-generalization.md)). |

---

## 1. What problem are we trying to solve?

After `predict()` we have 4,128 predictions and 4,128 real answers. We can't read 4,128 pairs of numbers and "get a feel" for it. We need **summaries** that answer:

- How wrong is the model on a typical area?
- Does it sometimes make *huge* mistakes?
- Is it better than just guessing the average?

---

## 2. The idea in plain English

Everything starts from one simple quantity: for each area, **how far was the guess from the truth?**

```text
error = actual value − predicted value
```

Then we summarise all those errors in different ways, and each way answers a slightly different question.

---

## 3. An everyday analogy

Imagine two delivery companies promising "delivery at 12:00".

- **Company A** is always about 10 minutes late.
- **Company B** is usually exactly on time, but once in a while it's 2 hours late.

Which is better? It depends on what you care about. On *average* they may be similar, but B's occasional disasters might be unacceptable. There's no single "best" way to summarise mistakes, so we use a few complementary ones.

---

## 4. Back to house prices

For each test area we compare the real `MedHouseVal` with the model's prediction:

```text
Actual     Prediction     Error

$300k      $280k          $20k    (guessed too low)
$400k      $450k          $50k    (guessed too high)
$200k      $210k          $10k    (guessed too high)
```

We'll use these three rows to compute every metric **by hand** first.

---

## 5. A tiny example — the metrics by hand

### MAE — the average size of a mistake

1. Take each error **without its sign** (we care how big it is, not which direction):

   ```text
   $20k, $50k, $10k
   ```

2. Average them:

   ```text
   MAE = ($20k + $50k + $10k) / 3 = $80k / 3 ≈ $26.7k
   ```

**Plain English:** *"On average, the prediction is off by about $26,700."*

MAE is the most beginner-friendly metric because it's in the **same units as the target**, and you can say it in one sentence.

### RMSE — large mistakes hurt more

1. **Square** each error:

   ```text
   20² = 400      50² = 2,500      10² = 100
   ```

2. Average the squares:

   ```text
   (400 + 2,500 + 100) / 3 = 3,000 / 3 = 1,000
   ```

3. Take the **square root** to get back to dollars:

   ```text
   RMSE = √1,000 ≈ $31.6k
   ```

Why do this? Squaring makes big errors **much** bigger: 50 → 2,500, but 10 → only 100. So RMSE is punished more by a few large mistakes.

Compare two models with the **same** MAE:

```text
Model A errors: $30k, $30k, $30k   → MAE = $30k   RMSE = $30k
Model B errors:  $0k,  $0k, $90k   → MAE = $30k   RMSE = √((0 + 0 + 8,100) / 3) = √2,700 ≈ $52k
```

MAE says they're equally good. RMSE reveals that Model B makes an occasional **big** mistake, like Company B from the analogy.

**Rule of thumb:** RMSE is always ≥ MAE. If RMSE is **much** larger than MAE, a few predictions are badly wrong, so go and look at them.

### R² — "how much better than guessing the average?"

R² compares the model with the **baseline** idea from [chapter 04](04-first-baseline-model.md).

1. The average of the actual values is ($300k + $400k + $200k) / 3 = **$300k**. If we always guessed $300k, the squared errors would be:

   ```text
   (300 − 300)² + (400 − 300)² + (200 − 300)² = 0 + 10,000 + 10,000 = 20,000
   ```

2. Our model's squared errors (from the RMSE calculation) add up to **3,000**.

3. R² asks: *what fraction of the "average-guesser's" squared error did the model get rid of?*

   ```text
   R² = 1 − (model's squared errors / average-guesser's squared errors)
      = 1 − 3,000 / 20,000
      = 1 − 0.15
      = 0.85
   ```

How to read R²:

| R² | Meaning |
|---|---|
| 1.0 | Perfect predictions |
| 0.85 | The model removed 85% of the squared error that "always guess the average" would make |
| 0.0 | No better than always guessing the average |
| below 0 | **Worse** than always guessing the average |

---

## 6. The official words

| Plain English | Official term |
|---|---|
| actual − predicted for one area | **Error** or **residual** |
| Average size of the mistakes | **MAE**: Mean Absolute Error |
| Square root of the average squared mistake | **RMSE**: Root Mean Squared Error |
| Fraction of variation explained, compared with guessing the average | **R²**: coefficient of determination ("R-squared") |
| A number used to judge a model | **Metric** |

---

## 7. A little Python

First, the tiny example, both by hand and with scikit-learn:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

actual = np.array([300, 400, 200])       # in $1,000s
predicted = np.array([280, 450, 210])

errors = actual - predicted               # [ 20, -50, -10]

mae_by_hand = np.abs(errors).mean()
rmse_by_hand = np.sqrt((errors ** 2).mean())
r2_by_hand = 1 - (errors ** 2).sum() / ((actual - actual.mean()) ** 2).sum()

print(mae_by_hand, mean_absolute_error(actual, predicted))       # 26.67 26.67
print(rmse_by_hand, root_mean_squared_error(actual, predicted))  # 31.62 31.62
print(r2_by_hand, r2_score(actual, predicted))                   # 0.85 0.85
```

Then on the real project, comparing the baseline and Linear Regression **on the same test set**:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame
X = df[["MedInc", "HouseAge", "AveRooms"]]
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, model in [
    ("Dummy", DummyRegressor(strategy="mean")),
    ("Linear Regression", LinearRegression()),
]:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{name:<18} MAE={mae:.3f} (≈${mae * 100_000:,.0f})  RMSE={rmse:.3f}  R²={r2:.3f}")
```

Fill in this table with your results when you reach this milestone:

| Model | Features | MAE | MAE in $ | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Dummy | 3 | | | | |
| Linear Regression | 3 | | | | |

### Don't stop at the numbers — look at the predictions

```python
import pandas as pd

model = LinearRegression().fit(X_train, y_train)

results = pd.DataFrame({
    "actual": y_test,
    "predicted": model.predict(X_test),
})
results["error"] = results["actual"] - results["predicted"]
results["abs_error"] = results["error"].abs()

print(results.head())                                    # a few typical rows
print(results.sort_values("abs_error").tail())           # the 5 worst predictions
```

---

## 8. The Python, line by line

| Code | What it does |
|---|---|
| `errors = actual - predicted` | One error per area. Positive = model guessed too low, negative = too high. |
| `np.abs(errors).mean()` | Remove the signs, then average → **MAE**. |
| `np.sqrt((errors ** 2).mean())` | Square, average, square-root → **RMSE**. |
| `((actual - actual.mean()) ** 2).sum()` | The squared error of "always guess the average". |
| `mean_absolute_error(y_true, y_pred)` | scikit-learn's MAE. The argument order is **(true, predicted)**. |
| `root_mean_squared_error(y_true, y_pred)` | scikit-learn's RMSE. |
| `r2_score(y_true, y_pred)` | scikit-learn's R². Here the order matters for the result, not just by convention. |
| `for name, model in [...]` | Train and evaluate every model in **exactly the same way** on **exactly the same split**. That's what makes the comparison fair. |
| `results.sort_values("abs_error").tail()` | Sort by size of mistake and show the biggest ones, so we can ask *what kind of areas does the model get wrong?* |

---

## 9. What happens inside (high level)

All three metrics are just arithmetic on the list of errors. The differences are in **what they emphasise**:

```text
                 units          sensitive to big mistakes?    compares with baseline?
MAE              $ (target)     no — every $ counts the same   no
RMSE             $ (target)     yes — squares them             no
R²               none (0–1)     yes — uses squared errors      yes, built in
```

### Why regression doesn't use "accuracy %"

In **classification** (spam / not spam), each prediction is simply right or wrong, so "95% accuracy" makes sense.

In **regression**, a prediction of $299,000 for an area worth $300,000 is excellent, but it's not *exactly* right. Almost no prediction will ever be exactly right. So "percentage correct" would be close to 0% even for a great model. Instead we ask **how far off** the predictions are, which is exactly what MAE and RMSE measure.

### R² is **not** "accuracy"

It's tempting to read R² = 0.60 as "60% accurate". It isn't. It means "the model removed 60% of the squared error that guessing the average would make". A model can have a decent R² and still be off by $50,000 on a typical area. Always report **MAE in dollars** alongside it.

### Error analysis

Metrics tell you *how much* the model is wrong. Looking at individual predictions tells you *where* and *why*:

- Are the biggest errors all in the capped $500k+ areas?
- Are they in very small or unusual areas (odd `AveOccup`)?
- Does the model consistently guess too low for expensive areas?

This is where improvement ideas come from ([chapter 09](09-ml-workflow.md)).

---

## 10. Common beginner mistakes

- **Reporting metrics without units.** "MAE = 0.53" means nothing to a reader; "≈ $53,000" does.
- **Treating R² as accuracy.** See above.
- **Using only one metric.** MAE can hide occasional disasters; RMSE and looking at the worst predictions reveal them.
- **Comparing models evaluated on different test sets.** Same `random_state`, same `X_test`/`y_test`, or the comparison is meaningless.
- **Not comparing with the baseline.** An MAE of $60k sounds bad until you see the baseline's is $150k (or good, until you see the baseline's is $62k).
- **Evaluating on training data.** Training metrics answer a different question (see [chapter 07](07-overfitting-and-generalization.md)).
- **Swapping the arguments** of `r2_score`. It must be `r2_score(y_true, y_pred)`.

---

## 11. Exercises

**Exercise 1 — By hand.**
Actual values: $100k, $200k, $300k, $400k. Predictions: $150k, $200k, $250k, $500k. Compute MAE and RMSE on paper. Which is larger, and why? Then check with scikit-learn.

<details>
<summary>Hint</summary>

The errors are −50, 0, +50, −100 (in $k). For MAE drop the signs; for RMSE square them first.
</details>

**Exercise 2 — Predict the R².**
What R² would the `DummyRegressor` get on the test set? Guess before running. (It's close to, but not exactly, a round number. Why not exactly?)

<details>
<summary>Hint</summary>

The dummy guesses the average of `y_train`, while R² compares against the average of `y_test`. These two averages are close but not identical.
</details>

**Exercise 3 — Fill the table.**
Complete the Dummy vs Linear Regression table above. By how many dollars does Linear Regression improve on the baseline's MAE?

**Exercise 4 — Hunt the worst errors.**
Show the 10 test areas with the largest absolute error. Join them back to `X_test` to see their features. What do they have in common?

<details>
<summary>Hint</summary>

`X_test.loc[worst.index]` selects the matching rows, because `results` keeps the same index as `X_test`.
</details>

---

## 12. Before continuing, I should be able to explain:

- □ What is a prediction error, and how is MAE computed from errors?
- □ Why is RMSE always at least as big as MAE, and what does a big gap between them suggest?
- □ What does R² = 0 mean? What about a negative R²?
- □ Why is R² not "accuracy", and why doesn't regression use an accuracy percentage?
- □ Why must all models be compared on the same test set?
- □ Why look at individual predictions instead of stopping at the metrics?

---

**Previous:** [← 05 — Linear Regression](05-linear-regression.md) · **Next:** [07 — Overfitting and Generalization →](07-overfitting-and-generalization.md)
