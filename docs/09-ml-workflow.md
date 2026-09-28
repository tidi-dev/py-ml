# 09 — The Complete ML Workflow: Putting It All Together

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → **09 Workflow** · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** This chapter connects everything. Read it now to see where the course is going, and come back to it after each milestone.

| Where we are | |
|---|---|
| **Earlier** | We met every piece separately: data, X/y, train/test split, baseline, Linear Regression, metrics, overfitting, Random Forest. |
| **Now** | See how those pieces fit into **one repeatable process**, and how to improve a model systematically instead of by trial and error. |
| **Next** | Apply this workflow to this project, one milestone at a time, and later to a completely different dataset. |

---

## 1. What problem are we trying to solve?

Knowing each tool isn't enough. Beginners often get stuck in a loop like this:

```text
try a model → score looks bad → try another model → try random settings
→ add some features → score changes → no idea why → try something else…
```

After a day of this, you have a number but no understanding. You can't explain *why* it's better, whether it will hold up on new data, or what to try next.

We need a **process** that makes every step purposeful and every result explainable.

---

## 2. The idea in plain English

The full supervised-learning workflow:

```text
Problem
   ↓
Collect/select data
   ↓
Understand data
   ↓
Define X and y
   ↓
Split data
   ↓
Baseline
   ↓
Train
   ↓
Predict
   ↓
Evaluate
   ↓
Analyze errors
   ↓
Improve
   ↓
Evaluate again
```

And the heart of it, the improvement loop:

```text
Change one thing
      ↓
Measure
      ↓
Compare
      ↓
Understand why
      ↓
(repeat)
```

ML is **iterative**: you go round the loop many times. Each lap should teach you something.

---

## 3. An everyday analogy

A good cook adjusting a recipe:

- tastes the dish first (**baseline**);
- changes **one** thing, e.g. a bit more salt (**one change**);
- tastes again (**measure**);
- compares with before (**compare**);
- thinks about *why* it's better or worse (**understand**);
- keeps a notebook of what they tried (**experiment log**).

If they changed the salt, the oven temperature and the cooking time all at once, they'd never know which change helped.

---

## 4. Back to house prices

Here's how each chapter maps onto the workflow for this project:

| Workflow step | In this project | Chapter |
|---|---|---|
| Problem | Predict median house value of a California block group | [00](00-ml-big-picture.md) |
| Collect/select data | `fetch_california_housing()` | [01](01-understanding-the-data.md) |
| Understand data | `head`, `info`, `describe`, caps, outliers | [01](01-understanding-the-data.md) |
| Define X and y | 3 features first, `MedHouseVal` as target | [02](02-features-and-target.md) |
| Split data | `train_test_split(test_size=0.2, random_state=42)` | [03](03-training-and-testing.md) |
| Baseline | `DummyRegressor(strategy="mean")` | [04](04-first-baseline-model.md) |
| Train / Predict | `LinearRegression().fit(...)`, `.predict(...)` | [05](05-linear-regression.md) |
| Evaluate | MAE (in $), RMSE, R² on the test set | [06](06-evaluating-a-model.md) |
| Analyze errors | Worst predictions, actual vs predicted, train vs test | [06](06-evaluating-a-model.md), [07](07-overfitting-and-generalization.md) |
| Improve | More features, a different model, feature ideas | [08](08-random-forest.md), this chapter |
| Evaluate again | Same split, same metrics, compare in a table | this chapter |

---

## 5. A tiny example — an experiment log

Here's what disciplined experimenting looks like. (The numbers are made up; you'll produce the real ones.)

| # | Model | Features | Train MAE | Test MAE | What changed, and why | What I learned |
|---|---|---|---:|---:|---|---|
| 1 | Dummy | 3 | 1.00 | 1.00 | Starting point | The bar to beat: ≈ $100k |
| 2 | Linear | 3 | 0.70 | 0.71 | Use features at all | Features clearly help |
| 3 | Linear | 8 | 0.62 | 0.63 | *Hypothesis:* location matters | Adding lat/long helped a lot |
| 4 | Random Forest | 8 | 0.15 | 0.45 | *Hypothesis:* relationships aren't straight lines | Better test MAE; big gap, so some overfitting |

Every row changes **one** thing compared with a previous row. Every row states **why**. That's what lets you explain your final model.

The questions to ask after each experiment:

> **What changed?**
> **Why did I change it?**
> **Did it actually improve generalization (test performance)?**

---

## 6. The official words

| Plain English | Official term |
|---|---|
| Going round the loop, improving step by step | **Iterative development** |
| A written record of what you tried and what happened | **Experiment log / tracking** |
| A specific, testable guess about what will help | **Hypothesis** |
| Creating new input columns from existing ones | **Feature engineering** |
| A third data split used to make choices, protecting the test set | **Validation set** |
| Evaluating on several different splits and averaging | **Cross-validation** |
| Chaining preparation steps and a model into one object | **Pipeline** |
| Transforming data before the model sees it | **Preprocessing** |
| Settings you choose (not learned), like `max_depth` | **Hyperparameters** |

---

## 7. A little Python

A tiny helper that makes every experiment **the same shape**: same split, same metrics, same output. You'll write something like this yourself in a later milestone.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame
y = df["MedHouseVal"]

# Fix the split ONCE, by row labels, so every experiment uses the same test rows.
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)


def run_experiment(model, features):
    X_train, X_test = df.loc[train_idx, features], df.loc[test_idx, features]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    return train_mae, test_mae


three = ["MedInc", "HouseAge", "AveRooms"]
eight = [c for c in df.columns if c != "MedHouseVal"]

for name, model, features in [
    ("Dummy", DummyRegressor(), three),
    ("Linear", LinearRegression(), three),
    ("Linear", LinearRegression(), eight),
]:
    train_mae, test_mae = run_experiment(model, features)
    print(f"{name:<7} {len(features)} features   train={train_mae:.3f}   test={test_mae:.3f}")
```

---

## 8. The Python, line by line

| Code | Why it's there |
|---|---|
| `train_test_split(df.index, ...)` | Splits the **row labels** once. Every experiment then selects the *same* rows, even with different feature sets. |
| `df.loc[train_idx, features]` | "These rows, these columns". `.loc` selects by label. |
| `def run_experiment(model, features)` | One function = one recipe. Every experiment is guaranteed to use the same split and the same metrics, so comparisons are fair. |
| returns `train_mae, test_mae` | Both numbers, so we can judge performance (test) *and* diagnose overfitting (the gap). |
| the `for` loop | Each entry is one row of the experiment log. Adding an experiment = adding one line. |

---

## 9. What happens inside (high level)

### Why "change one thing"?

If you change the model *and* the features at once and the score improves, you don't know which change helped, or whether one helped and the other hurt. Changing one thing at a time turns each experiment into a small, clear lesson.

### Protecting the test set: validation sets and cross-validation (later)

Every time you look at the test score and make a decision because of it, the test set becomes a little less "unseen" ([chapter 07](07-overfitting-and-generalization.md)). Over many experiments, you can accidentally tune your choices to that particular test set.

The more rigorous setup, for later in the project:

```text
All data
  ├── Training data  ──► used for learning AND for choosing between options
  │                      (via a validation split or cross-validation)
  └── Test data      ──► touched ONCE, at the very end
```

**Cross-validation** is a way of getting a more reliable score from the training data alone. You split it into, say, 5 parts. You train on 4 parts and evaluate on the 5th, rotate so each part is used for evaluation once, and average the 5 scores. That tells you whether a result is solid or just lucky with one particular split. scikit-learn's tool for this is `cross_val_score`. You'll learn it after train/test splitting feels completely natural.

### Pipelines (later)

Some models need the data prepared first, e.g. rescaling features. A **pipeline** glues "prepare the data" and "train the model" into one object with one `fit()` and one `predict()`. Its main benefit is **correctness**: the preparation step learns only from the training data, which prevents a subtle kind of data leakage. It also makes the model easier to reproduce and save. In this project we'll introduce `Pipeline` only when there's a real preparation step that needs it, not just to use it.

### Feature engineering (later)

For any new feature you invent, first answer three questions:

1. *Why might this contain predictive information?*
2. *Would this information actually be available at prediction time?* (leakage check, [chapter 02](02-features-and-target.md))
3. *Did test performance actually improve?*

---

## 10. Common beginner mistakes

- **Randomly trying models** without a hypothesis. Every experiment should answer a question.
- **Changing several things at once.** You can't tell what helped.
- **Comparing experiments that used different test data.** Fix the split once.
- **Not writing anything down.** Two days later you won't remember what "run 7" was.
- **Optimising the score instead of understanding.** A slightly better number you can't explain is worth less than a slightly worse one you understand.
- **Tuning against the test set** until it looks good. Use validation / cross-validation for choices, and keep the test set for the final check.
- **Assuming more features always help.** Test it: it's experiment 3 vs experiment 2 in the log.
- **Leaking target information** through engineered features or data preparation done before the split.

---

## 11. Exercises

**Exercise 1 — Map the workflow.**
Without looking, write the 13 workflow steps from memory. Next to each, write one sentence about why it exists.

**Exercise 2 — Design an experiment.**
Write a hypothesis about `Latitude` and `Longitude` (e.g. "location matters a lot for house value"). Describe the **two** experiments that would test it, and what result would support or reject it. Which single thing differs between the two experiments?

<details>
<summary>Hint</summary>

Same model, same split, same everything, except one experiment includes the two location columns and the other doesn't.
</details>

**Exercise 3 — Start your own experiment log.**
Create a table like the one in section 5, in a Markdown file or a notebook cell. Fill it in as you complete each milestone of the project.

**Exercise 4 — Transfer.**
Pick a different prediction problem (e.g. predicting a used car's price). Walk through the 13 workflow steps for it. What would `X`, `y` and the baseline be? What leakage risks can you think of?

---

## 12. Before continuing, I should be able to explain:

- □ What are the main steps of the supervised-learning workflow, and why does each exist?
- □ Why is ML iterative, and why change only one thing per experiment?
- □ What should every experiment record, and what three questions should I ask after each?
- □ Why must every compared experiment use the same test set?
- □ Why is repeatedly tuning against the test set a problem, and what is cross-validation for?
- □ How would I apply this same workflow to a completely different dataset?

---

**Previous:** [← 08 — Random Forest](08-random-forest.md) · **Back to the start:** [README](../README.md) · [Glossary](glossary.md)
