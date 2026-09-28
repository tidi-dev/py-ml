# 04 — Your First Baseline Model: The "Always Guess the Average" Strategy

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → **04 Baseline** → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** The project code doesn't do this yet. Read it now for orientation, or when you reach this milestone. Do the exercises when you get here in the project.

| Where we are | |
|---|---|
| **Earlier** | We split the data into a training set (to learn from) and a test set (to check with) ([chapter 03](03-training-and-testing.md)). |
| **Now** | Build the simplest possible "model" and measure how wrong it is. That becomes the bar every real model must beat. |
| **Next** | Build our first *real* model, Linear Regression, and see whether it beats this bar ([chapter 05](05-linear-regression.md)). |

---

## 1. What problem are we trying to solve?

Suppose you train a model and it's wrong by **$60,000** on average. Is that good or bad?

You can't tell. A number on its own has no meaning. You need something to **compare** it with.

So before building anything clever, we ask:

> How well could we do **without learning anything useful at all**?

---

## 2. The idea in plain English

The laziest reasonable strategy for guessing house values is:

> **Ignore all the information and always guess the average house value.**

It's not a good strategy. But it's *a* strategy, it takes no intelligence, and it gives us a number to beat.

If our "smart" model can't do clearly better than this, it isn't providing much value, however sophisticated it sounds.

---

## 3. An everyday analogy

Before judging whether our smart prediction system is useful, compare it with **someone who always guesses the average house value**.

A weather forecaster who says *"tomorrow will be the same as the average day in this month"* is right surprisingly often. A fancy new forecasting system that is *no better* than that isn't worth paying for. The simple strategy is the **baseline**, the "do nothing clever" reference point.

---

## 4. Back to house prices

Suppose the average house value in our training data is **$300k** (made-up round number). The baseline predicts:

```text
Area A → $300k
Area B → $300k
Area C → $300k
... every area → $300k
```

It doesn't matter whether the area is a rich coastal neighbourhood or a poor rural one. The guess is always the same.

Every future model will be judged against this:

```text
Baseline error:            (some number, e.g. $100k)
Linear Regression error:   must be clearly lower
Random Forest error:       must be clearly lower
```

---

## 5. A tiny example

**Training set** (the answers the baseline learns from):

```text
Area    Value
A       $200k
B       $300k
C       $400k
        ─────
mean  = ($200k + $300k + $400k) / 3 = $300k
```

"Training" the baseline means computing that one number: **$300k**.

**Test set** (new areas the baseline hasn't seen):

```text
Area    Actual    Baseline prediction    How wrong?
D       $250k     $300k                  $50k
E       $500k     $300k                  $200k
F       $250k     $300k                  $50k
```

Average mistake = ($50k + $200k + $50k) / 3 = **$100k**.

That average mistake is called the **Mean Absolute Error (MAE)**. We'll study it properly in [chapter 06](06-evaluating-a-model.md). Now we know: *any model worth using on this data should be wrong by clearly less than $100k on average.*

Notice how badly the baseline does on area E, an expensive area. It can't tell expensive areas apart from cheap ones, because it doesn't look at any features.

---

## 6. The official words

| Plain English | Official term |
|---|---|
| A simple reference strategy that every real model must beat | **Baseline** |
| scikit-learn's ready-made "don't learn anything useful" model | **`DummyRegressor`** |
| Always guessing the average of the training answers | `strategy="mean"` |
| The average size of the mistakes | **Mean Absolute Error (MAE)** |

A `DummyRegressor` is still a real scikit-learn model: it has `fit()` and `predict()` just like the clever ones. That means we can compare it with other models using exactly the same code.

---

## 7. A little Python

```python
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame
X = df[["MedInc", "HouseAge", "AveRooms"]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

mae = mean_absolute_error(y_test, baseline_predictions)
print(f"Baseline MAE: {mae:.2f}  (≈ ${mae * 100_000:,.0f})")
```

(When you reach this milestone, **predict** the MAE before running this. Roughly how far is a typical area from the average value? Look back at `describe()` in [chapter 01](01-understanding-the-data.md).)

---

## 8. The Python, line by line

### `baseline = DummyRegressor(strategy="mean")`

Creates an **untrained** model and tells it which lazy strategy to use: the mean (average). Other strategies exist, e.g. `"median"`.

### `baseline.fit(X_train, y_train)`

```text
X_train  = information about the training areas  → the dummy IGNORES it
y_train  = correct answers for those areas       → the dummy averages them
fit()    = "learn from these examples"           → for the dummy, just: compute the mean
```

After this line, the model has stored exactly **one number**: the average of `y_train` (about 2.07, i.e. ≈ $207,000). You can see it with `baseline.constant_`.

Why pass `X_train` at all if it's ignored? So that `DummyRegressor` has the **same interface** as every other model. You can swap it for `LinearRegression()` without changing any other line.

### `baseline_predictions = baseline.predict(X_test)`

For each of the 4,128 test areas, return the stored average. The result is an array of 4,128 identical numbers.

Note that we predict on **`X_test`**, the areas the model has never seen, and **not** `X_train`.

### `mean_absolute_error(y_test, baseline_predictions)`

Compares each true answer with its prediction, takes the size of each mistake (ignoring whether it's too high or too low), and averages them. The order is **(true answers, predictions)**.

### `mae * 100_000`

The target is in units of $100,000, so we multiply to get dollars. An MAE of 0.40 would mean ≈ **$40,000**. Always report results in units a human understands.

---

## 9. What happens inside (high level)

```text
fit:      y_train ──► average ──► store one number
predict:  any X   ──► return the stored number, once per row
```

That's all. There are no patterns, no features, no learning about the relationship between `X` and `y`. That's exactly what makes it useful as a reference: its error measures **how hard the problem is when you know nothing**.

When a real model beats it, the difference tells you how much the model actually **learned from the features**:

```text
Baseline MAE − Model MAE  =  value added by learning from X
```

---

## 10. Common beginner mistakes

- **Skipping the baseline.** Then you have no idea whether your model's score is good. A model with an impressive-looking error may be barely better than guessing the average.
- **Thinking the baseline is a "real" solution.** It's a measuring stick, not a product.
- **Computing the baseline on the test set.** The average must come from `y_train`. Using `y_test.mean()` would leak test information into the baseline.
- **Comparing against a baseline on a different split.** The baseline and the real model must be evaluated on the **same** `X_test`/`y_test`.
- **Forgetting the units.** An MAE of `0.9` is not "90 cents". It's about $90,000.

---

## 11. Exercises

**Exercise 1 — Predict before you run.**
Using the `describe()` output from chapter 01, guess what `baseline.constant_` will be. Then guess the baseline MAE. Then run the code and compare.

<details>
<summary>Hint</summary>

The constant is the mean of `y_train`, which should be close to the mean of the whole `MedHouseVal` column. For the MAE, think about how far a *typical* value is from the average.
</details>

**Exercise 2 — Mean vs median.**
Try `DummyRegressor(strategy="median")`. Is its MAE higher or lower than the mean strategy's? Can you think of a reason why?

<details>
<summary>Hint</summary>

Remember that `MedHouseVal` has a long tail of expensive areas (chapter 01). Which of mean and median is pulled more by that tail?
</details>

**Exercise 3 — Features don't matter here.**
Train the baseline with 3 features and then with all 8. Does the MAE change? Explain why.

**Exercise 4 — Write it without scikit-learn.**
Using only pandas, compute the baseline predictions and the MAE by hand: take `y_train.mean()`, subtract it from every `y_test` value, take absolute values, and average. Check you get the same MAE.

---

## 12. Before continuing, I should be able to explain:

- □ What is a baseline, and why do we need one before training a real model?
- □ What does `DummyRegressor(strategy="mean")` learn during `fit()`?
- □ Why does the baseline predict the same value for every area?
- □ What does an MAE of 0.40 mean in dollars?
- □ Why must the baseline's average come from `y_train`, not `y_test`?

---

**Previous:** [← 03 — Training and Testing](03-training-and-testing.md) · **Next:** [05 — Linear Regression →](05-linear-regression.md)
