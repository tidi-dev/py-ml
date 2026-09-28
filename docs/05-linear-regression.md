# 05 — Linear Regression: Drawing the Best Line Through the Data

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → **05 Linear regression** → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** The project code doesn't do this yet. Read it now for orientation, or when you reach this milestone. Do the exercises when you get here in the project.

| Where we are | |
|---|---|
| **Earlier** | We built a baseline that always guesses the average. It ignores the features completely ([chapter 04](04-first-baseline-model.md)). |
| **Now** | Build a model that actually **uses** the features: Linear Regression. |
| **Next** | Measure properly how good it is, and whether it beats the baseline ([chapter 06](06-evaluating-a-model.md)). |

---

## 1. What problem are we trying to solve?

The baseline gives every area the same guess. But we saw in [chapter 01](01-understanding-the-data.md) that areas differ a lot: some are rich, some are poor, some are on the coast. Surely we can do better by **using** that information.

The simplest way to use information is to find a **general trend**. For example: *"the higher the income, the higher the value, by roughly this much."*

---

## 2. The idea in plain English

Start with **one** feature: median income.

If you plotted every neighbourhood as a dot, with income along the bottom and house value up the side, you'd see a cloud of dots that generally rises from left to right:

```text
House Value
    ↑
    |                    *   *
    |              *  *    *
    |         *  *   *  *
    |      *   *  *
    |   *  * *
    | *  *
    +--------------------------→ Income
```

Linear Regression draws **one straight line** through that cloud, the line that fits the general trend as well as possible:

```text
House Value
    ↑
    |                    * ./*
    |              *  * ./ *
    |         *  *  ./*  *
    |      *   * ./*
    |   *  * *./
    | *  * ./
    +--------------------------→ Income
```

Once we have the line, predicting is easy: find the area's income on the bottom axis, go up to the line, and read off the value.

A straight line is fully described by just **two numbers**:

- **where it starts** (its height when income is zero);
- **how steep it is** (how much value goes up for each extra unit of income).

Training is finding the two numbers that make the line fit the dots best.

---

## 3. An everyday analogy

A taxi fare:

```text
fare = $3 starting fee  +  $2 per kilometre
```

If you took many taxi rides and wrote down each distance and fare, you could work out the starting fee and the per-kilometre price **from the receipts alone**, without anyone telling you the pricing rules. That's what Linear Regression does: it recovers "starting fee" and "price per unit" from examples.

Real house prices are messier than taxi fares (the dots don't fall exactly on a line), so the model finds the line that is **as close as possible to most of the dots**.

---

## 4. Back to house prices

With one feature:

```text
predicted value  =  starting point  +  (effect per unit of income) × income
```

With several features, we simply add more effects:

```text
prediction
=
starting point
+
effect of income
+
effect of house age
+
effect of rooms
...
```

Each feature gets **its own number** that says how much the prediction changes when that feature goes up by one unit, **keeping the other features the same**.

---

## 5. A tiny example

### One feature

Three made-up areas (income in $10,000s, value in $1,000s):

```text
Income   Value
  2      $150k
  4      $250k
  6      $350k
```

Each extra 1 unit of income (+$10,000) adds $50k of value, and a line through these points would start at $50k. So:

```text
value = $50k + $50k × income
```

Prediction for a new area with income 5:

```text
value = $50k + $50k × 5 = $50k + $250k = $300k
```

### Several features

Now suppose training found these (made-up) numbers, with the target in units of $100k as in our data:

```text
starting point            = 0.50
effect per unit of income = 0.40
effect per year of age    = 0.01
effect per extra room     = 0.05
```

For **Area A** from chapter 00 (income = 5, age = 20, rooms = 5):

```text
prediction = 0.50
           + 0.40 × 5     = 2.00
           + 0.01 × 20    = 0.20
           + 0.05 × 5     = 0.25
           ─────────────────────
           = 2.95   →  ≈ $295,000
```

(Area A's real value was $250k, so this guess would be $45k too high.)

That's all a trained Linear Regression model does when it predicts: **multiply each feature by its number and add everything up.**

---

## 6. The official words

Only now, with the idea clear, here's the standard notation:

```text
y = b0 + b1·x1 + b2·x2 + b3·x3 + ...
```

| Symbol | Plain English | In our example |
|---|---|---|
| `y` | the predicted value | predicted `MedHouseVal` |
| `x1`, `x2`, `x3` | the feature values for one area | income, house age, rooms |
| `b0` | the starting point: the prediction if every feature were 0 | 0.50 |
| `b1`, `b2`, `b3` | how much the prediction changes per unit of each feature | 0.40, 0.01, 0.05 |
| `·` | multiply | |

Official names:

| Plain English | Official term |
|---|---|
| The starting point `b0` | **Intercept** (or bias) |
| The per-feature numbers `b1, b2, …` | **Coefficients** (or weights) |
| The numbers the model learns during training | **Parameters** |
| Predicting a number with a weighted sum of features | **Linear Regression** |
| The gap between the real value and the prediction | **Residual** (or error) |

"Linear" means each feature's effect is a **straight line**: every extra unit of income adds the *same* amount, whether income goes from 1 to 2 or from 10 to 11.

---

## 7. A little Python

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame
features = ["MedInc", "HouseAge", "AveRooms"]
X = df[features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("intercept:", model.intercept_)
print(pd.Series(model.coef_, index=features))
```

And to prove there's no magic, here's one prediction done by hand:

```python
first_area = X_test.iloc[0]
by_hand = model.intercept_ + (first_area * model.coef_).sum()

print("by hand:     ", by_hand)
print("by predict():", predictions[0])
```

The two numbers are the same.

---

## 8. The Python, line by line

### `model = LinearRegression()`

Creates an **untrained** model. It knows the *form* of the rule (starting point + weighted sum of features), but the numbers are unknown.

### `model.fit(X_train, y_train)`

```text
X_train = information about the 16,512 training areas
y_train = the correct value for each of them
fit()   = find the intercept and coefficients that make
          the predictions on these areas as close as possible
          to the correct values
```

After `fit()`, the model stores **4 numbers**: 1 intercept + 3 coefficients. It does **not** store the 16,512 training rows. That's why it isn't a database of answers.

### `predictions = model.predict(X_test)`

For each of the 4,128 test areas: multiply each feature by its coefficient, add them up, add the intercept. The result is an array of 4,128 predicted values. Nothing is learned here. It just applies the rule.

### `model.intercept_` and `model.coef_`

The learned numbers. The trailing underscore is a scikit-learn convention: **attributes ending in `_` only exist after `fit()`**, because they are *learned from data*.

`model.coef_` is an array in the **same order as the columns of `X`**. Wrapping it in `pd.Series(..., index=features)` labels each number with its feature name.

### `(first_area * model.coef_).sum()`

Multiplies each feature value by its coefficient (element by element) and adds the results. It's exactly the "by hand" calculation from section 5.

---

## 9. What happens inside (high level)

How does `fit()` find the "best" line?

1. Any line makes some mistakes: for each training area, the gap between the real value and the line's value.
2. Linear Regression scores a line by **squaring each gap and adding them all up**. Squaring makes every gap positive and makes big gaps count much more than small ones.
3. The best line is the one with the **smallest total of squared gaps**. This is called **least squares**.

You could imagine trying millions of lines and keeping the best. Fortunately, for straight lines mathematicians found a **shortcut**: a formula that computes the best intercept and coefficients directly, in one step. scikit-learn uses that formula, which is why `fit()` is almost instant.

### What a coefficient does (and doesn't) tell you

A coefficient of, say, 0.40 for `MedInc` means:

> If two areas are identical in every other feature used by the model, but one has median income 1 unit ($10,000) higher, the model predicts its value to be 0.40 ($40,000) higher.

Three warnings:

- **Units matter.** `Population` is counted in people (typically about 1,000 per area), while `MedInc` is in $10,000s (typically about 3–5). A tiny coefficient on a big-unit feature can still have a large effect. You can't compare coefficient sizes directly across features with different units.
- **Coefficients depend on the other features.** Add or remove a feature and the others can change, sometimes even flip sign, because features that move together "share" the effect.
- **Correlation, not causation.** The model says "areas with higher X tend to have higher y". It doesn't say that raising X would *cause* y to rise.

### Limitation: it can only draw straight lines

If the true relationship curves (value rises quickly at first, then levels off), a straight line can't follow it. That's the motivation for tree-based models in [chapter 08](08-random-forest.md).

---

## 10. Common beginner mistakes

- **Thinking `fit()` stores the training answers.** It stores only the intercept and one coefficient per feature.
- **Reading coefficients as "importance".** Their size depends on units, and on which other features are in the model.
- **Believing coefficients prove causation.** They describe patterns in *this* data.
- **Taking the intercept literally.** It's the prediction when *every* feature is 0. With all 8 features, that means income 0, zero rooms, and latitude/longitude 0 (a point in the Atlantic Ocean). No real area looks like that, so the intercept can be a strange number. It's a mathematical starting point, not a meaningful price.
- **Forgetting to beat the baseline.** Linear Regression isn't automatically good. Compare its test error with the `DummyRegressor` from [chapter 04](04-first-baseline-model.md) on the **same** test set.
- **Evaluating on `X_train`.** Predictions on training data look better than they will on new data.
- **Assuming the line fits everyone.** A single straight line can be quite wrong for unusual areas (e.g. the capped $500k+ ones).

---

## 11. Exercises

**Exercise 1 — Predict the signs.**
Before fitting, write down whether you expect each coefficient (`MedInc`, `HouseAge`, `AveRooms`) to be positive or negative, and why. Then fit the model and compare. Is any sign surprising? Can you think of a reason?

<details>
<summary>Hint</summary>

Remember that each coefficient means "the effect of this feature **when the other features stay the same**". Areas with more rooms also tend to have higher incomes. Once income is already accounted for, what extra information do rooms add?
</details>

**Exercise 2 — One prediction by hand.**
Pick any row of `X_test`. Using a calculator and the printed intercept and coefficients, compute the prediction by hand. Check it against `model.predict`.

**Exercise 3 — One feature only.**
Train a Linear Regression using only `MedInc`. Print its intercept and coefficient. Write the model as a sentence: "value ≈ ___ + ___ × income".

<details>
<summary>Hint</summary>

`X` must still be a DataFrame: `df[["MedInc"]]` (double brackets).
</details>

**Exercise 4 — Count the parameters.**
How many numbers does a Linear Regression store when trained on all 8 features? How does that compare to the number of training rows?

---

## 12. Before continuing, I should be able to explain:

- □ In plain English, what does Linear Regression look for in the data?
- □ What are the intercept and the coefficients, and where do they come from?
- □ What exactly does `predict()` compute for one area?
- □ What does a coefficient of 0.40 on `MedInc` mean, in dollars?
- □ Why can't I compare coefficient sizes directly to decide which feature is "most important"?
- □ What kind of relationship can a straight line *not* capture?

---

**Previous:** [← 04 — Your First Baseline Model](04-first-baseline-model.md) · **Next:** [06 — Evaluating a Model →](06-evaluating-a-model.md)
