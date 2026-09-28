# 07 — Overfitting and Generalization: Memorising vs Understanding

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → **07 Overfitting** → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** The project code doesn't do this yet. Read it now for orientation, or when you reach this milestone. Do the exercises when you get here in the project.

| Where we are | |
|---|---|
| **Earlier** | We learned to measure errors with MAE, RMSE and R², always on the **test** set ([chapter 06](06-evaluating-a-model.md)). |
| **Now** | Compare errors on the **training** set with errors on the **test** set, and learn what the gap between them tells us. |
| **Next** | Meet a more flexible model, Random Forest, which is powerful but prone to exactly this problem ([chapter 08](08-random-forest.md)). |

---

## 1. What problem are we trying to solve?

Suppose you build a model and it scores a near-perfect error on the training data. Great, right?

Not necessarily. Remember the goal from [chapter 00](00-ml-big-picture.md): we want good predictions for **areas the model has never seen**. A model can be excellent on its training data and still be poor on new data.

> **The goal is not to perform perfectly on training data.**
> **The goal is to perform well on new data.**

---

## 2. The idea in plain English

A model can learn two kinds of things from its training examples:

1. **Real patterns** that also hold for new areas ("higher income → higher value").
2. **Accidental details** that only happen to be true for these particular examples. That includes noise, coincidences and one-off quirks.

Learning (1) helps on new data. Learning (2) *looks* helpful on training data but does nothing, or harm, on new data.

- A model that learns too much of (2) is **overfitting**: it has memorised instead of understood.
- A model too simple to even learn (1) is **underfitting**: it hasn't learned enough.
- The sweet spot learns the real patterns and ignores the quirks. That's **good generalization**.

---

## 3. An everyday analogy

Three students prepare for a maths exam using a book of 100 practice questions with answers.

| Student | How they study | Practice score | Exam score (new questions) |
|---|---|---|---|
| **The memoriser** | Memorises all 100 answers word for word | 100% | Poor: the questions are different |
| **The skimmer** | Reads a few pages, learns almost nothing | Poor | Poor |
| **The learner** | Works out *how* to solve each kind of question | Very good (not perfect) | Very good |

- The memoriser is **overfitting**: brilliant on what they've seen, lost on anything new.
- The skimmer is **underfitting**: bad everywhere.
- The learner **generalises**.

The practice score alone can't tell you which student is which. Only the exam can.

---

## 4. Back to house prices

For every model we now compute the error **twice**:

```text
Train MAE  = error on the examples the model learned from
Test MAE   = error on examples it never saw
```

Then we read the pair:

```text
very low training error
+
much higher test error
        ↓
OVERFITTING (the memoriser)


high training error
+
similar high test error
        ↓
UNDERFITTING (the skimmer)


reasonably low training error
+
test error not much higher
        ↓
GOOD GENERALIZATION (the learner)
```

---

## 5. A tiny example

Imagine three models on our data (made-up numbers, in $100k):

```text
Model           Train MAE   Test MAE    Gap      Diagnosis
─────────────────────────────────────────────────────────────
Dummy             1.00        1.00      0.00     underfitting: learned nothing
Model P           0.60        0.62      0.02     learns something, generalises well
Model Q           0.00        0.50      0.50     memorised the training set
```

Which is best? Not Model Q because of its perfect training score. **Look at the test column.** Model Q has the lowest test error, so it *is* the most useful of the three on new data, even though it's clearly overfitting. The gap tells us something else: it has room to improve, because a less "memorising" version might do even better on the test set.

Two lessons:

1. **Choose models by test performance, not training performance.**
2. **A big train/test gap is a warning sign** that the model is spending effort memorising, and that it might generalise better if constrained.

---

## 6. The official words

| Plain English | Official term |
|---|---|
| How well the model does on its own training examples | **Training performance** (train error) |
| How well it does on held-out examples | **Test performance** (test error) |
| Doing well on new, unseen data | **Generalization** |
| Learning quirks of the training data that don't carry over | **Overfitting** (high variance) |
| Being too simple to capture the real patterns | **Underfitting** (high bias) |
| How flexible a model is, i.e. how complicated the patterns it can learn | **Model complexity** / **capacity** |

---

## 7. A little Python

We'll use a model we'll properly meet in [chapter 08](08-random-forest.md): the **decision tree**. For now, all you need to know is that it learns by asking yes/no questions about the features, and `max_depth` limits **how many questions in a row** it may ask. That's a dial from "very simple" to "extremely flexible".

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for depth in [1, 2, 4, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, tree.predict(X_train))
    test_mae = mean_absolute_error(y_test, tree.predict(X_test))

    print(f"max_depth={depth!s:>4}   train MAE={train_mae:.3f}   test MAE={test_mae:.3f}")
```

When you reach this milestone, **predict before you run**: for which depth do you expect the lowest train MAE? The lowest test MAE? Are they the same depth?

---

## 8. The Python, line by line

| Code | What it does |
|---|---|
| `fetch_california_housing(return_X_y=True, as_frame=True)` | A shortcut that returns `X` (all 8 features) and `y` directly. |
| `for depth in [1, 2, 4, 8, 12, None]` | Train six trees, from very simple (1 question) to unlimited (`None` = keep asking questions until every training area is perfectly separated). |
| `DecisionTreeRegressor(max_depth=depth, random_state=42)` | `max_depth` controls complexity. `random_state` makes the result repeatable. |
| `tree.predict(X_train)` | Predictions on the **training** data, used *only* to diagnose overfitting, never to choose the model. |
| `tree.predict(X_test)` | Predictions on the **test** data, the honest estimate. |
| `f"{depth!s:>4}"` | Formatting only: `!s` turns `None` into text, `>4` right-aligns it. |

You'll see something like:

```text
depth 1      train error high      test error high         ← underfitting
depth 4-8    train error lower     test error lower        ← better
depth None   train error 0.000!    test error goes UP      ← overfitting
```

With unlimited depth, the tree creates a separate "leaf" for almost every training area. It has **memorised** the training set, and its training error drops to zero. But its test error is *worse* than the medium-depth tree's.

---

## 9. What happens inside (high level)

Every real dataset contains **noise**: variation that no feature can explain. Two areas with identical features can have different values for reasons that aren't in the data (a new school, a view of the sea, a data-entry error).

- A **flexible** model has enough freedom to bend itself around every noisy point. That drives the training error towards zero, but noise doesn't repeat in new data, so the bending doesn't help on the test set.
- A **rigid** model (like a single straight line) can't bend enough to follow even the real curves. It underfits.

```text
Error                                          ● = test error
  ↑                                            ○ = training error
  │ ●○
  │    ●                                  ●
  │     ○  ●                          ●
  │          ○  ●               ●
  │                ●   ●   ●
  │                ○
  │                     ○
  │                          ○
  │                               ○
  │                                    ○   ○
  └─────────────────────────────────────────────→ Model complexity
    underfitting        sweet spot       overfitting
```

The training error (○) keeps falling as the model gets more flexible. The test error (●) falls at first, then **rises again** once the model starts memorising.

That's why the **test set** exists ([chapter 03](03-training-and-testing.md)): training error alone would always tell you "more complex is better".

### How do you fight overfitting?

You don't need these yet, but it helps to know they exist:

- limit model complexity (e.g. `max_depth`);
- use more training data;
- use fewer, better features;
- average many different models (the idea behind Random Forest in [chapter 08](08-random-forest.md));
- evaluate more reliably with **cross-validation** ([chapter 09](09-ml-workflow.md)).

---

## 10. Common beginner mistakes

- **Assuming a lower training error means a better model.** The unlimited tree has the *lowest possible* training error and is not the best model.
- **Only computing test error.** You then can't *diagnose* overfitting. Compute both, and compare.
- **Choosing settings (like `max_depth`) by looking at the test score over and over.** This slowly "fits" the test set, so it stops being unseen. The proper tool is a **validation set** or **cross-validation** ([chapter 09](09-ml-workflow.md)).
- **Thinking overfitting means the model is useless.** An overfit model can still beat simpler ones on test data. The gap just says it might be improved.
- **Thinking complex models always overfit and simple models never do.** It depends on the data size, the noise and the settings. **Measure, don't assume.**

---

## 11. Exercises

**Exercise 1 — Predict, then run.**
Before running the code in section 7, sketch on paper what you think the train MAE and test MAE will look like as depth goes from 1 to unlimited. Then run it. Where were you wrong?

**Exercise 2 — Find the sweet spot.**
Try every depth from 1 to 20. Which depth gives the lowest *test* MAE? Why is choosing the depth this way (looking at test scores) slightly cheating?

<details>
<summary>Hint</summary>

If you choose a setting because it scores best on the test set, then the test set has influenced your model. Is it still "never seen"? (Chapter 09 introduces the fix.)
</details>

**Exercise 3 — Linear Regression's gap.**
Compute train MAE and test MAE for the `LinearRegression` from chapter 05. Is the gap large or small? What does that suggest about whether Linear Regression is overfitting or underfitting?

**Exercise 4 — Explain it to a friend.**
Using the student analogy, explain in three sentences why a model with 0 training error can be a worse model.

---

## 12. Before continuing, I should be able to explain:

- □ What is the difference between training performance and test performance?
- □ What is overfitting, and what train/test pattern reveals it?
- □ What is underfitting, and what pattern reveals it?
- □ What does "generalization" mean, and why is it the real goal?
- □ Why is a lower training error not necessarily a better model?
- □ Why shouldn't I choose model settings by repeatedly checking the test set?

---

**Previous:** [← 06 — Evaluating a Model](06-evaluating-a-model.md) · **Next:** [08 — Random Forest →](08-random-forest.md)
